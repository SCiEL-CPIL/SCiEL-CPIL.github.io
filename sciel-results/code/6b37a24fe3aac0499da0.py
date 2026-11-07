import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Exploit symmetry: h[i] = h[N-1-i]
    # Optimize the first half (25 variables)
    
    def objective(h_half):
        h = np.zeros(N)
        h[:N//2] = h_half
        h[N//2:] = h_half[::-1]
        
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 0.0
        h = h / s
        
        g = np.convolve(h, h)
        m = len(g)
        
        y = np.zeros(m + 2)
        y[1:m+1] = g
        
        w = 1.0 / (m + 1)
        sum_val = 2.0 * np.sum(y**2) + np.sum(y[:-1] * y[1:])
        l2 = (w / 3.0) * sum_val
        
        norm_1 = np.sum(g) / (m + 1)
        
        # Use high p for better L_infinity approximation
        p = 100
        if np.max(g) < 1e-12:
             return 0.0
        norm_inf = np.sum(g**p) ** (1.0/p)
        
        denom = norm_1 * norm_inf
        if denom < 1e-12:
            return 0.0
            
        val = l2 / denom
        return -val

    bounds = [(0, None)] * (N//2)
    
    starts = []
    # 1. Uniform
    starts.append(np.ones(N//2) / N)
    
    # 2. Gaussians
    x = np.linspace(-1, 1, N//2)
    for sigma in [0.2, 0.5, 0.8, 1.0]:
        g = np.exp(-x**2 / (2*sigma**2))
        starts.append(g / np.sum(g))
        
    # 3. Random
    np.random.seed(42)
    for _ in range(10):
        r = np.random.rand(N//2)
        starts.append(r / np.sum(r))
        
    best_val = -np.inf
    best_h_half = None
    
    for h0 in starts:
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_h_half = res.x
        except Exception:
            continue
            
    if best_h_half is None:
        return list(np.ones(N) / N)
        
    best_h = np.zeros(N)
    best_h[:N//2] = best_h_half
    best_h[N//2:] = best_h_half[::-1]
    
    best_h = np.maximum(best_h, 0)
    best_h /= np.sum(best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
