import numpy as np
import time
from scipy.optimize import minimize

def _compute_entropies(P):
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    def entropy(vals, P):
        flat = P.flatten()
        vflat = vals.flatten()
        unique_vals, inverse = np.unique(vflat, return_inverse=True)
        probs = np.zeros(len(unique_vals))
        np.add.at(probs, inverse, flat)
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    H_1 = entropy(X - Y, P)
    H_0 = entropy(X, P)
    H_1p = entropy(X + Y, P)
    H_2 = entropy(X + 2*Y, P)
    return H_1, H_0, H_1p, H_2

def _score(P):
    H_1, H_0, H_1p, H_2 = _compute_entropies(P)
    M = max(H_0, H_1p, H_2)
    if M < 1e-9:
        return 0.0
    return H_1 / M

def arithmetic_kakeya_distribution():
    start_time = time.time()
    deadline = start_time + 570
    n = 10
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = _score(best_P)
    
    # Phase 1: Diagonal-symmetric optimization (20 vars: a[d+9] for d=-9..9)
    def diag_to_P(a):
        P = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P[y, x] = a[x - y + 9]
        s = P.sum()
        if s < 1e-15:
            return np.ones((n, n)) / (n * n)
        return P / s
    
    def diag_objective(a):
        a = np.clip(a, 1e-15, None)
        P = diag_to_P(a)
        return -_score(P)
    
    diag_inits = [
        np.ones(20),
        np.array([1.0 / (10 - abs(d)) for d in range(-9, 10)]),
        np.random.default_rng(42).uniform(0.5, 2.0, 20),
        np.random.default_rng(123).uniform(0.1, 5.0, 20),
        np.random.default_rng(7).uniform(0.1, 10.0, 20),
    ]
    
    for a0 in diag_inits:
        if time.time() > deadline - 90:
            break
        try:
            res = minimize(diag_objective, a0, method='Nelder-Mead',
                          options={'maxiter': 15000, 'xatol': 1e-10, 'fatol': 1e-12})
            a = np.clip(res.x, 1e-15, None)
            P = diag_to_P(a)
            s = _score(P)
            if s > best_score:
                best_score = s
                best_P = P.copy()
        except Exception:
            pass
    
    # Phase 2: Full 100-var refinement
    if time.time() < deadline - 30:
        theta0 = np.log(best_P + 1e-15)
        def full_objective(theta):
            P = np.exp(theta)
            P = P / P.sum()
            return -_score(P)
        
        try:
            res = minimize(full_objective, theta0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-11})
            P = np.exp(res.x)
            P = P / P.sum()
            s = _score(P)
            if s > best_score:
                best_score = s
                best_P = P.copy()
        except Exception:
            pass
    
    # Final validation
    best_P = np.clip(best_P, 0, None)
    if best_P.sum() < 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / best_P.sum()
    
    return best_P
