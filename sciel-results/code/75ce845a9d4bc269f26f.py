"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def construct():
    dx = 0.5 / N  # = 0.01
    
    def compute_ratio(heights):
        a = np.array(heights, dtype=np.float64)
        if len(a) == 0 or np.sum(a) < 1e-15:
            return 0.0
        norm2_sq = np.sum(a**2) * dx
        norm1 = np.sum(a) * dx
        
        # Compute autoconvolution using np.correlate
        autoconv = np.correlate(a, a, mode='full') * dx
        sup_autoconv = np.max(autoconv)
        
        if sup_autoconv < 1e-15 or norm1 < 1e-15:
            return 0.0
        
        ratio = norm2_sq / (norm1 * sup_autoconv)
        return ratio
    
    def neg_ratio(heights):
        return -compute_ratio(heights)
    
    best_ratio = 0.0
    best_heights = [1.0] * N
    
    # Try various structural families
    # Family 1: Single spike
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        r = compute_ratio(list(h))
        if r > best_ratio:
            best_ratio = r
            best_heights = list(h)
    
    # Family 2: Two adjacent spikes
    for pos in range(N - 1):
        h = np.zeros(N)
        h[pos] = 1.0
        h[pos+1] = 1.0
        r = compute_ratio(list(h))
        if r > best_ratio:
            best_ratio = r
            best_heights = list(h)
    
    # Family 3: Use differential evolution to optimize
    if time.time() - START_TIME < TIME_LIMIT - 30:
        bounds = [(0.0, 10.0)] * N
        # Use a subset of parameters to speed up
        result = differential_evolution(
            neg_ratio, bounds, maxiter=50, popsize=15,
            tol=1e-4, seed=42, polish=False
        )
        if result.fun < 0:
            r = -result.fun
            if r > best_ratio:
                best_ratio = r
                best_heights = list(result.x)
    
    # Local search refinement
    if time.time() - START_TIME < TIME_LIMIT - 15:
        current = np.array(best_heights, dtype=np.float64)
        best_r = best_ratio
        step_size = 0.5
        for iteration in range(300):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            improved = False
            for _ in range(5):
                candidate = current.copy()
                n_perturb = np.random.randint(1, 4)
                for _ in range(n_perturb):
                    pos = np.random.randint(0, N)
                    delta = np.random.uniform(-step_size, step_size)
                    candidate[pos] = max(0.0, candidate[pos] + delta)
                r = compute_ratio(list(candidate))
                if r > best_r:
                    best_r = r
                    best_heights = list(candidate)
                    current = candidate
                    improved = True
            if not improved:
                step_size *= 0.9
            else:
                step_size = min(step_size * 1.1, 2.0)
    
    best_heights = [max(0.0, h) for h in best_heights]
    if np.sum(best_heights) < 1e-10:
        best_heights = [1.0] * N
    
    return best_heights
