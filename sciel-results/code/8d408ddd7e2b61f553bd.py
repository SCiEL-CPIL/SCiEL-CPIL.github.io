import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    # Objective to maximize: ratio = ||f||_2^2 / (||f||_1 * ||f*f||_inf)
    # Scale invariance: normalize ||f||_1 = 1 inside objective.
    
    def objective(f):
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
        return -(norm2_sq / max_conv)

    bounds = [(0, 1) for _ in range(N)]
    
    best_val = -np.inf
    best_f = np.ones(N) / N
    
    start_time = time.time()
    time_limit = 110.0
    
    # New Approach: Asymmetric seeds
    # Symmetric functions yield ratio 1. Asymmetric functions can yield > 1.
    # We use exponential ramps, linear ramps, and skewed distributions.
    
    guesses = []
    i = np.arange(N)
    
    # 1. Exponential ramps (geometric progressions)
    for r in [1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 0.9, 0.8, 0.5, 0.33]:
        if abs(r - 1.0) < 1e-9:
            g = np.ones(N)
        else:
            g = r ** i
        g = g / np.sum(g)
        guesses.append(g)
        # Reverse ramp
        guesses.append(g[::-1].copy())
        
    # 2. Linear ramps
    ramp = i.copy()
    ramp = ramp / np.sum(ramp)
    guesses.append(ramp)
    guesses.append(ramp[::-1].copy())
    
    # 3. Skewed Gaussians
    for alpha in [0.5, 1.0, 2.0, 4.0]:
        for shift in [-15, -10, -5, 5, 10, 15]:
            center = (N - 1) / 2.0 + shift
            g = np.exp(-alpha * (i - center)**2)
            g = g / np.sum(g)
            guesses.append(g)
            
    # 4. "Comb" or sparse asymmetric
    for gap in [2, 3, 5]:
        f = np.zeros(N)
        f[::gap] = 1.0
        non_zero_idx = np.arange(0, N, gap)
        f[non_zero_idx] = np.linspace(0.1, 1.0, len(non_zero_idx))
        f = f / np.sum(f)
        guesses.append(f)
        
    # 5. Random asymmetric
    np.random.seed(42)
    for _ in range(20):
        r = np.random.rand(N)
        # Apply a random skew power
        power = np.random.choice([2, 3, 0.5, 0.2])
        r = r ** power
        r = r / np.sum(r)
        guesses.append(r)

    # Run Powell optimization for each guess
    for x0 in guesses:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(objective, x0, method='Powell', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.fun < best_val:
                best_val = res.fun
                best_f = res.x
        except Exception:
            pass

    if best_val == -np.inf:
        best_f = np.ones(N) / N
        
    best_f = np.maximum(best_f, 0)
    s = np.sum(best_f)
    if s > 0:
        best_f = best_f / s
        
    return list(best_f)
