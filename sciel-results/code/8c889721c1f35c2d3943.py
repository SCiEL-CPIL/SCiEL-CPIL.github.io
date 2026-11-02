import numpy as np
from scipy.optimize import minimize
import time

N = 600

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    S = np.zeros(N)
    for k in range(N):
        S[k] = np.dot(h[:N-k], h[k:])
    H = np.sum(h)
    if H < 1e-6:
        return np.inf
    return np.max(S) / H

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try several parameterized families to get good initial guesses
    best_score = np.inf
    best_h = np.ones(N)
    
    idx = np.arange(N)
    center = (N - 1) / 2.0
    
    # Family 1: Uniform
    h = np.ones(N)
    s = compute_score(h)
    if s < best_score:
        best_score = s
        best_h = h.copy()
    
    # Family 2: Triangle (linear taper)
    h = np.clip(1 - np.abs(idx - center) / center, 0, 1)
    if np.max(h) > 0:
        h = h / np.max(h)
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Family 3: Various Gaussians
    for sigma in [N/8, N/6, N/5, N/4, N/3, N/2.5, N/2]:
        h = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
        h = h / np.max(h)
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Family 4: Power-law taper
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        h = np.clip(1 - (np.abs(idx - center) / center) ** p, 0, 1)
        if np.max(h) > 0:
            h = h / np.max(h)
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    # Family 5: Two-hump
    for sep in [N/4, N/3, N/2.5]:
        h1 = np.exp(-0.5 * ((idx - center + sep) / (N/10)) ** 2)
        h2 = np.exp(-0.5 * ((idx - center - sep) / (N/10)) ** 2)
        h = h1 + h2
        if np.max(h) > 0:
            h = h / np.max(h)
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    # Now use SLSQP to optimize from the best initial guess
    # Formulation: minimize max_k S_k subject to sum(h) = 1, h >= 0
    # Equivalent to: minimize t s.t. S_k <= t for all k, sum(h)=1, h>=0
    # But simpler: just minimize max_k S_k with sum(h)=1 constraint
    
    # Normalize best_h to sum 1
    h_start = best_h / np.sum(best_h)
    
    def objective(h):
        S = np.zeros(N)
        for k in range(N):
            S[k] = np.dot(h[:N-k], h[k:])
        return np.max(S)
    
    def gradient(h):
        S = np.zeros(N)
        for k in range(N):
            S[k] = np.dot(h[:N-k], h[k:])
        k_max = np.argmax(S)
        g = np.zeros(N)
        for i in range(N):
            if i + k_max < N:
                g[i] += h[i + k_max]
            if i - k_max >= 0:
                g[i] += h[i - k_max]
        return g
    
    # Use a smooth approximation for better convergence
    def smooth_objective(h, beta=20.0):
        S = np.zeros(N)
        for k in range(N):
            S[k] = np.dot(h[:N-k], h[k:])
        m = np.max(S)
        if m < 1e-12:
            return 0.0
        return m + np.log(np.sum(np.exp(beta * (S - m)))) / beta
    
    def smooth_gradient(h, beta=20.0):
        S = np.zeros(N)
        for k in range(N):
            S[k] = np.dot(h[:N-k], h[k:])
        m = np.max(S)
        if m < 1e-12:
            return np.zeros(N)
        exp_S = np.exp(beta * (S - m))
        weights = exp_S / np.sum(exp_S)
        # Gradient of smooth max is weighted sum of gradients of each S_k
        g = np.zeros(N)
        for k in range(N):
            w = weights[k]
            if w < 1e-10:
                continue
            for i in range(N):
                if i + k < N:
                    g[i] += w * h[i + k]
                if i - k >= 0:
                    g[i] += w * h[i - k]
        return g
    
    # Try both exact and smooth versions
    for use_smooth in [False, True]:
        if time.time() > deadline - 30:
            break
        
        if use_smooth:
            obj = lambda h: smooth_objective(h)
            jac = lambda h: smooth_gradient(h)
        else:
            obj = objective
            jac = gradient
        
        cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)})
        bounds = [(0, None)] * N
        
        try:
            res = minimize(obj, h_start, jac=jac, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 300, 'ftol': 1e-10})
            if res.x is not None:
                h_opt = np.clip(res.x, 0, None)
                if np.sum(h_opt) > 1e-6:
                    h_opt = h_opt / np.sum(h_opt)
                    s = compute_score(h_opt)
                    if s < best_score:
                        best_score = s
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)})
        bounds = [(0, None)] * N
        try:
            res = minimize(smooth_objective, h_start, jac=smooth_gradient, method='SLSQP',
                          bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
            if res.x is not None:
                h_opt = np.clip(res.x, 0, None)
                if np.sum(h_opt) > 1e-6:
                    h_opt = h_opt / np.sum(h_opt)
                    s = compute_score(h_opt)
                    if s < best_score:
                        best_score = s
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
