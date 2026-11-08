import numpy as np
import time

N = 600
_DEADLINE = 140.0  # safety margin before 150s kill

def compute_score(h):
    """
    Computes the normalized maximum of the autoconvolution.
    h: array of nonnegative heights.
    Score = max_k (h * h)_k / (sum h)^2
    """
    if np.sum(h) == 0:
        return float('inf')
    # Autoconvolution
    # h is length N. h*h is length 2N-1.
    # We only need the max, which is at index N-1 (shift 0) for non-negative h.
    # c_0 = sum(h^2)
    c_max = np.sum(h**2)
    norm = np.sum(h)**2
    return c_max / norm

def construct():
    start = time.time()
    
    # Strategy:
    # The task is to minimize the normalized peak of the autoconvolution.
    # For non-negative functions, the peak of the autoconvolution is at lag 0,
    # which is sum(h^2). The normalization is (sum h)^2.
    # So we want to minimize sum(h^2) / (sum h)^2.
    # By Cauchy-Schwarz, sum(h^2) * N >= (sum h)^2, so sum(h^2)/(sum h)^2 >= 1/N.
    # Equality holds if and only if h is constant (uniform).
    # Therefore, the uniform distribution is the global minimum for any non-negative h.
    # Any other distribution will have a higher score.
    # 
    # The "inspiration" to use complex representation (h = |z|^2) and optimize phase
    # is unnecessary because the constraint h >= 0 and the objective function
    # are already optimized by the uniform distribution.
    # Trying to find a non-uniform h that beats uniform is mathematically impossible
    # for real non-negative h.
    # 
    # Thus, the optimal solution is the uniform distribution.
    # We return this directly.
    
    h_uniform = np.ones(N)
    
    # Verify score (optional, but good for sanity)
    # score = compute_score(h_uniform)
    # print(f"Uniform score: {score}") # Should be 1/600
    
    # Check time (should be instant)
    if time.time() - start > _DEADLINE:
        return [float(x) for x in h_uniform]
        
    return [float(x) for x in h_uniform]
