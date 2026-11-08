import numpy as np
import scipy.optimize as opt
import time

N = 50
M = 2 * N - 1
W = 1.0 / (M + 1)

def compute_value(h):
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        return 0.0
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        return 0.0
    if np.sum(h) == 0:
        return 0.0
        
    g = np.convolve(h, h)
    # g length is M
    # y = [0, g, 0]
    # l2 sum over i=0..M of (y_i^2 + y_i y_{i+1} + y_{i+1}^2)
    
    # Vectorized calculation of sum
    # y_i are 0, g_0, ..., g_{M-1}, 0
    # Terms:
    # i=0: 0 + 0 + g_0^2 = g_0^2
    # i=M: g_{M-1}^2 + 0 + 0 = g_{M-1}^2
    # i=k (1<=k<=M-1): g_{k-1}^2 + g_{k-1}g_k + g_k^2
    # Wait, the sum is i=0 to M.
    # y has indices 0 to M+1.
    # y[0]=0, y[1]=g[0], ..., y[M]=g[M-1], y[M+1]=0.
    # Term i uses y[i], y[i+1].
    # i=0: y[0], y[1] -> 0, g[0] -> g[0]^2
    # i=1: y[1], y[2] -> g[0], g[1] -> g[0]^2 + g[0]g[1] + g[1]^2
    # ...
    # i=M: y[M], y[M+1] -> g[M-1], 0 -> g[M-1]^2
    
    # Sum = g[0]^2 + g[M-1]^2 + sum_{k=1}^{M-1} (g[k-1]^2 + g[k-1]g[k] + g[k]^2)
    # This is sum_{k=0}^{M-1} g[k]^2 + sum_{k=0}^{M-2} g[k]g[k+1] + sum_{k=0}^{M-1} g[k]^2
    # = 2 * sum(g^2) + sum(g[:-1] * g[1:])
    
    sum_g_sq = np.sum(g * g)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    
    l2_sum = 2 * sum_g_sq + sum_g_g_next
    l2 = (W / 3.0) * l2_sum
    
    S = np.sum(h)
    norm_1 = (S * S) / (M + 1)
    norm_inf = np.sum(h * h) # g[0]
    
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def neg_value(h):
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
            
    # Initializations
    rng = np.random.RandomState(42)
    starts = []
    starts.append(np.ones(N))
    starts.append(np.exp(-0.5 * (np.linspace(0, N-1, N) - N/2)**2 / (N/4)**2))
    starts.append(np.sqrt(np.maximum(0, 1 - ((2*np.linspace(0, N-1, N)/(N-1)) - 1)**2)))
    
    for _ in range(10):
        starts.append(rng.uniform(0, 1, N))
        
    bounds = [(0, None)] * N
    
    for s in starts:
        if time.time() > deadline:
            break
        x0 = np.maximum(0.0, s)
        try:
            res = opt.minimize(
                neg_value,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            track(res.x)
        except Exception:
            pass
            
    # Polish with tighter tolerances
    for _ in range(20):
        if time.time() > deadline:
            break
        pert = best_h + rng.normal(0, 0.05, N)
        pert = np.maximum(0, pert)
        try:
            res = opt.minimize(
                neg_value,
                pert,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            track(res.x)
        except Exception:
            pass
            
    return best_h.tolist()
