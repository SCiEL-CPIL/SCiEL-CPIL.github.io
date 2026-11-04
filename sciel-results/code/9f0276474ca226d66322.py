"""Second autocorrelation inequality - optimize 50 nonnegative heights."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def autoconv(h):
    """Compute full autoconvolution of h."""
    n = len(h)
    result = np.zeros(2*n - 1)
    for i in range(n):
        result[i:i+n] += h[i] * h
    return result

def score_fn(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    ac = autoconv(h)
    ac_max = np.max(np.abs(ac))
    if ac_max < 1e-15:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    deadline = time.time() + 140  # safety margin
    
    best_score = 0.0
    best_h = np.ones(N)
    
    # Try various structured patterns first
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Sparse: alternating
    for k in range(1, N+1):
        for start in range(N):
            h = np.zeros(N)
            for i in range(k):
                idx = (start + i * (N // k)) % N
                h[idx] = 1.0
            if np.sum(h) > 0:
                candidates.append(h)
    
    # Random sparse patterns
    np.random.seed(42)
    for trial in range(200):
        k = np.random.randint(1, N+1)
        h = np.zeros(N)
        idx = np.random.choice(N, size=k, replace=False)
        h[idx] = 1.0
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        s = score_fn(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    if time.time() > deadline - 30:
        return list(best_h)
    
    # Local optimization from best candidate
    def neg_score(x):
        return -score_fn(x)
    
    def neg_score_jac(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(N):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            g[i] = (neg_score(xp) - neg_score(xm)) / (2*eps)
        return g
    
    # Try optimization from best candidate
    x0 = best_h.copy()
    # Scale to avoid issues
    x0 = x0 / np.max(x0) if np.max(x0) > 0 else x0
    
    try:
        res = minimize(neg_score, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < -best_score:
            best_score = -res.fun
            best_h = res.x
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        return list(np.maximum(best_h, 0))
    
    # Try from a few more starting points
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        np.random.seed(100 + trial)
        x0 = np.random.rand(N)
        x0 = x0 / np.max(x0)
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_score:
                best_score = -res.fun
                best_h = res.x
        except Exception:
            pass
    
    return list(np.maximum(best_h, 0))
