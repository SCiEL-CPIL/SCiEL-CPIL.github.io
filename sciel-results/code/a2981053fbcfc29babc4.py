import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Normalize to remove scale invariance for optimization stability
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    autoconv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(autoconv)
    if max_conv == 0:
        return 0.0
    return l2_sq / max_conv

def _objective(x):
    """Objective function to minimize (negative ratio). x are log-heights."""
    h = np.exp(x)
    # Optional: normalize h to prevent overflow/underflow in exp, though ratio is scale invariant
    # But for Nelder-Mead, keeping scale roughly fixed helps.
    # The ratio function normalizes internally, so x scale doesn't matter for value, 
    # but it affects the geometry of the search space.
    # We can normalize x to have mean 0 to reduce the flat direction.
    x_centered = x - np.mean(x)
    h = np.exp(x_centered)
    return -_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin
    
    best_score = 0.0
    best_h = np.ones(N) / N  # Default: uniform
    
    # Define a few heuristic starting points
    starts = []
    
    # 1. Uniform
    starts.append(np.zeros(N))
    
    # 2. Triangular
    tri = np.arange(N)
    tri = np.minimum(tri, N-1-tri)
    starts.append(np.log(tri + 1e-6))
    
    # 3. Gaussian
    g = np.exp(-((np.arange(N) - N/2)**2) / (N/10))
    starts.append(np.log(g + 1e-6))
    
    # 4. Random sparse (few non-zeros)
    for _ in range(5):
        mask = np.random.rand(N) > 0.8
        if not np.any(mask): mask[0] = True
        h_rand = np.where(mask, np.random.rand(N), 0)
        if np.sum(h_rand) > 0:
             starts.append(np.log(h_rand + 1e-6))

    # 5. Random dense
    for _ in range(10):
        h_rand = np.random.rand(N)
        starts.append(np.log(h_rand + 1e-6))

    # 6. Alternating
    alt = np.ones(N)
    alt[::2] = 0.1
    starts.append(np.log(alt))

    # Run optimization from each start
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            # The result x is the log-heights.
            # We need to convert back to h.
            # Since the objective normalized x, the resulting h might be scaled arbitrarily.
            # We just need the shape.
            x_res = res.x
            # Center x_res to avoid huge/small numbers
            x_res = x_res - np.mean(x_res)
            h_res = np.exp(x_res)
            
            # Normalize h_res to be a valid candidate (non-negative)
            # The ratio is scale invariant, so we can return any positive scaling.
            # Let's normalize sum to 1 for consistency, or just return h_res.
            # The task asks for "nonnegative step heights". Any scaling is fine.
            # I will return h_res normalized to max 1.0 to keep numbers small.
            h_res = h_res / np.max(h_res)
            
            score = _ratio(h_res)
            
            if score > best_score:
                best_score = score
                best_h = h_res.copy()
                
        except Exception:
            continue
            
    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
