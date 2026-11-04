import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import time

N = 50

def construct():
    def ratio(f):
        f = np.maximum(f, 0)
        s = np.sum(f)
        if s == 0:
            return 0
        f_norm = f / s
        norm2_sq = np.sum(f_norm**2)
        conv = np.convolve(f_norm, f_norm, mode='full')
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0
        return norm2_sq / max_conv

    best_val = -np.inf
    best_f = np.ones(N) / N
    
    start_time = time.time()
    time_limit = 110.0

    # Approach: Parametric search over Geometric Decays
    # f_i proportional to r^i for i=0..K-1, 0 elsewhere.
    
    def make_geometric(K, r):
        f = np.zeros(N)
        if K <= 0:
            return f
        if K > N:
            K = N
        # Use log-space to handle r near 0 or 1? 
        # r in (0, 1).
        if r <= 0:
            f[0] = 1.0
            return f
        if r >= 1.0:
            f[:K] = 1.0
            return f
        indices = np.arange(K)
        f[:K] = r ** indices
        s = np.sum(f)
        if s > 0:
            f /= s
        return f

    # Optimize r for various K
    for K in [5, 10, 20, 30, 40, 50]:
        if time.time() - start_time > time_limit:
            break
        
        def objective_r(log_r):
            # log_r in [-10, 0] corresponds to r in [4e-5, 1]
            r = np.exp(log_r)
            f = make_geometric(K, r)
            return -ratio(f)
        
        try:
            # Use minimize_scalar on log_r
            res = minimize_scalar(objective_r, bounds=(-15, 0), method='bounded', options={'xatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_f = make_geometric(K, np.exp(res.x))
        except Exception:
            pass

    # Also try a Gaussian bump, just in case
    def make_gaussian(width):
        i = np.arange(N)
        center = (N - 1) / 2.0
        g = np.exp(-((i - center) / width) ** 2)
        s = np.sum(g)
        if s > 0:
            g /= s
        return g

    def objective_w(log_w):
        w = np.exp(log_w)
        f = make_gaussian(w)
        return -ratio(f)

    try:
        res_g = minimize_scalar(objective_w, bounds=(-2, 2), method='bounded', options={'xatol': 1e-6})
        if res_g.fun < best_val:
            best_val = res_g.fun
            best_f = make_gaussian(np.exp(res_g.x))
    except Exception:
        pass

    # Finalize
    if best_val == -np.inf:
        best_f = np.ones(N) / N
        
    best_f = np.maximum(best_f, 0)
    s = np.sum(best_f)
    if s > 0:
        best_f = best_f / s
        
    return list(best_f)
