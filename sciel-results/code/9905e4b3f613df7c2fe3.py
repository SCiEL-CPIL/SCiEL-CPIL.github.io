import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 50

def evaluate(h):
    """Compute the value for a given set of step heights."""
    h = np.array(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Define bounds: heights between 0 and 10
    bounds = [(0, 10)] * N
    
    # Use differential evolution, a global optimization method
    # This is different from local gradient-based methods
    # It uses population-based search which can escape local minima
    
    try:
        result = differential_evolution(
            fun=lambda h: -evaluate(h),
            bounds=bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            init='latinhypercube',
            callback=lambda x, convergence: (time.time() > deadline)
        )
        
        best_h = result.x
        best_val = evaluate(best_h)
        
    except Exception as e:
        # Fallback to a simple random search if DE fails
        best_h = np.ones(N)
        best_val = evaluate(best_h)
        
        for _ in range(500):
            if time.time() > deadline:
                break
            h = np.random.uniform(0, 5, N)
            val = evaluate(h)
            if val > best_val:
                best_val = val
                best_h = h
    
    if best_h is None or not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
