import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start = time.time()
    deadline = start + 110  # 120s limit, leave 10s margin

    def overlaps(g):
        # Compute overlaps for all shifts k
        # f[k] = sum_{i=0}^{N-1-k} g[i] * (1 - g[i+k])
        f = np.zeros(N)
        for k in range(N):
            # Vectorized dot product
            f[k] = np.dot(g[:N-k], 1.0 - g[k:])
        return f

    def raw_value(g):
        return float(np.max(overlaps(g)) * (2.0 / N))

    def project(g):
        # Project g to [0,1]^N with sum N/2
        # 1. Clip to [0,1]
        gp = np.clip(g, 0.0, 1.0)
        # 2. Adjust sum
        target = N / 2.0
        current_sum = np.sum(gp)
        diff = target - current_sum
        
        # Find indices that are not at bounds
        free = (gp > 1e-9) & (gp < 1.0 - 1e-9)
        if np.any(free):
            # Distribute diff among free indices
            gp[free] += diff / np.sum(free)
            gp = np.clip(gp, 0.0, 1.0)
            # If clipping changed the sum again, do a second pass
            current_sum = np.sum(gp)
            diff = target - current_sum
            free = (gp > 1e-9) & (gp < 1.0 - 1e-9)
            if np.any(free) and np.abs(diff) > 1e-12:
                gp[free] += diff / np.sum(free)
                gp = np.clip(gp, 0.0, 1.0)
        else:
            # If no free indices, we are stuck at bounds. 
            # This is rare for sum N/2. 
            # If we are stuck, we must move some bound variables.
            # This case is complex. For robustness, if this happens, 
            # we might return a default or try a random perturbation.
            # Given N=100 and target 50, it's unlikely to be stuck 
            # unless g is all 0s or 1s. 
            # If g is all 0.5, free is all True.
            pass
            
        # Final check: if sum is still off, force adjustment on the largest free set or just return
        # For safety, if the sum is not close to target, return a constant 0.5 sequence
        if abs(np.sum(gp) - target) > 1e-6:
             return np.full(N, 0.5)
             
        return gp

    def make_obj(g):
        f = overlaps(g)
        # Use a soft max to make it smooth
        tau = 0.1
        m = np.max(f)
        soft_max_val = m + tau * np.log(np.sum(np.exp((f - m) / tau)))
        # Add penalty for constraint violation (though project handles it, 
        # adding a small penalty helps optimizer stay in feasible region)
        penalty = 10.0 * (np.sum(g) - N / 2.0) ** 2
        return soft_max_val * (2.0 / N) + penalty

    # Initial guess: A cosine wave ensures some structure
    x = np.arange(N)
    g0 = 0.5 + 0.3 * np.cos(2 * np.pi * x / N)
    g0 = project(g0)
    
    best_g = g0
    best_val = raw_value(best_g)
    
    # Run optimization
    try:
        res = minimize(make_obj, g0, method='L-BFGS-B',
                       bounds=[(0.0, 1.0)] * N,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        g_opt = project(res.x)
        val_opt = raw_value(g_opt)
        if val_opt < best_val:
            best_g = g_opt
            best_val = val_opt
    except Exception:
        pass

    # Local search polish if time remains
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            f = overlaps(best_g)
            k_max = int(np.argmax(f))
            
            # Gradient of f[k_max]
            # f[k] = sum_{i=0}^{N-1-k} g[i] - g[i]g[i+k]
            # df/dg[j] = 1 (if j < N-k) - g[j+k] (if j < N-k) - g[j-k] (if j >= k)
            grad = np.zeros(N)
            mask1 = np.arange(N) < (N - k_max)
            grad[mask1] += 1.0
            grad[mask1] -= g_opt[mask1 + k_max] if hasattr(g_opt, 'mask1') else best_g[mask1 + k_max] # Fix: use best_g
            # Correct gradient calculation:
            grad = np.zeros(N)
            mask1 = np.arange(N) < (N - k_max)
            grad[mask1] += 1.0
            grad[mask1] -= best_g[mask1 + k_max]
            mask2 = np.arange(N) >= k_max
            grad[mask2] -= best_g[mask2 - k_max]
            
            # Line search
            for step in [0.01, 0.005, 0.001, 0.0005]:
                g_new = project(best_g - step * grad)
                val_new = raw_value(g_new)
                if val_new < best_val:
                    best_g = g_new
                    best_val = val_new
                    break
    
    # Final safety check
    best_g = project(best_g)
    if np.sum(best_g) < 1e-9:
        return np.full(N, 0.5)
        
    return best_g
