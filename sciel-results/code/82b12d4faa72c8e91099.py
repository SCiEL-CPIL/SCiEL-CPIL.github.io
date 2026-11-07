"""Optimized program for second autocorrelation inequality."""
import numpy as np
import time

N = 50
_DEADLINE = time.time() + 140.0  # Safety margin before 150s kill

def _compute_value(h):
    """Compute the objective value for given heights."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)  # spacing: total range 1.0 over m+1 intervals
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1] * y[1:m+2] + y[1:m+2]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def construct():
    """Return 50 nonnegative step heights maximizing the objective."""
    best_h = [1.0] * N
    best_val = _compute_value(best_h)
    
    # Start with a peaked shape: Gaussian centered
    # The key is to make g[0] (peak) large relative to the rest of g
    # A sharp peak in h gives a sharp peak in g
    
    # Try various peaked profiles
    candidates = []
    
    # Gaussian with various widths
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        x = np.arange(N)
        center = (N - 1) / 2.0
        h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        h[h < 1e-10] = 0.0
        candidates.append(h)
    
    # Triangular / tent
    for width in [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]:
        x = np.arange(N)
        center = (N - 1) / 2.0
        h = np.maximum(0, 1 - np.abs(x - center) / width)
        candidates.append(h)
    
    # Power-law falloff
    for p in [1, 2, 3, 4, 5, 8, 12, 20]:
        x = np.arange(N)
        center = (N - 1) / 2.0
        h = np.maximum(0, 1 - np.abs(x - center) / (N / 2.0)) ** p
        candidates.append(h)
    
    # Uniform (seed)
    candidates.append(np.ones(N))
    
    # Single spike
    for spike_pos in range(N):
        h = np.zeros(N)
        h[spike_pos] = 1.0
        candidates.append(h)
    
    # Two spikes
    for d in range(1, N):
        h = np.zeros(N)
        h[N//2 - d//2] = 1.0
        h[N//2 - d//2 + d] = 1.0
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > _DEADLINE:
            break
        val = _compute_value(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    # Local optimization: gradient-free refinement around best
    if time.time() < _DEADLINE - 10:
        # Nelder-Mead style local search
        from scipy.optimize import minimize
        
        def neg_value(x):
            return -_compute_value(x)
        
        # Try a few starting points near the best
        starts = [best_h.copy()]
        # Add small perturbations
        for _ in range(3):
            pert = best_h.copy() * (1 + 0.1 * np.random.randn(N))
            pert[pert < 0] = 0
            starts.append(pert)
        
        for start in starts:
            if time.time() > _DEADLINE - 5:
                break
            try:
                res = minimize(neg_value, start, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < -best_val:
                    h_opt = res.x
                    h_opt[h_opt < 0] = 0
                    val_opt = _compute_value(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final: ensure nonnegative and finite
    best_h = np.maximum(0, best_h)
    best_h[~np.isfinite(best_h)] = 0
    
    return list(best_h)
