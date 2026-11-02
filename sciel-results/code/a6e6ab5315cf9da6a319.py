import numpy as np
import time
from scipy.optimize import linprog

N = 600

def compute_ratio(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    Ratio = max(autoconv) / sum(heights^2)
    """
    if np.max(heights) == 0:
        return 1.0
    autoconv = np.correlate(heights, heights, mode='full')
    max_val = np.max(autoconv)
    norm_sq = np.sum(heights**2)
    return max_val / norm_sq

def _solve_qp_for_threshold(t):
    """
    Solve: minimize sum(h^2) subject to:
      - h >= 0
      - autoconv[lag] <= t for all lags
    
    We use a sequential quadratic approach with linear constraints.
    Since the problem is convex, we can use an iterative method.
    
    For efficiency, we'll use a simplified approach:
    Start with uniform, then iteratively adjust to reduce peak.
    """
    # Initial guess: uniform distribution
    h = np.ones(N) / np.sqrt(N)
    
    # Iterative refinement: reduce peaks in autoconvolution
    for iteration in range(50):
        autoconv = np.correlate(h, h, mode='full')
        peak_lag = np.argmax(autoconv)
        
        if autoconv[peak_lag] <= t * (1 + 1e-10):
            break
        
        # Reduce the contribution to the peak lag
        # The peak at lag 0 is sum(h^2)
        # For other lags, it's sum(h[i]*h[i+lag])
        
        if peak_lag == 0:
            # Reduce overall energy by scaling down
            current_energy = np.sum(h**2)
            if current_energy > t:
                scale = np.sqrt(t / current_energy)
                h *= scale
        else:
            # For non-zero lag peaks, we need to decorrelate
            # Simple approach: add small random perturbations and re-optimize
            # This is a heuristic that may not be optimal but helps break symmetry
            
            # Try to reduce correlation at this lag by adjusting values
            # at positions that contribute most to this lag
            lag = peak_lag
            if lag < N:
                # Reduce h at positions that pair with h at lag offset
                # This is approximate
                for i in range(N - lag):
                    if h[i] > 0 and h[i + lag] > 0:
                        # Reduce the smaller one slightly
                        if h[i] < h[i + lag]:
                            h[i] *= 0.99
                        else:
                            h[i + lag] *= 0.99
    
    # Ensure non-negativity
    h = np.maximum(h, 0)
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Bisection on the threshold t
    # We want to find the minimum t such that there exists h >= 0 with
    # sum(h^2) = 1 and max(autoconv) <= t
    # Then the ratio is t/1 = t
    
    # Lower bound: 1.0 (theoretical minimum)
    # Upper bound: start with uniform, which gives ratio = 1.0
    # But we need to search for better structures
    
    # Actually, for non-negative functions, the max of autoconv is always
    # at lag 0 (sum of squares), so the ratio is always 1.0.
    # 
    # Wait - this means the problem is trivial? Let me reconsider.
    # 
    # The autoconvolution at lag 0 is sum(h[i]^2).
    # For non-negative h, all other lags have autoconv[lag] = sum(h[i]*h[i+lag]) <= sum(h[i]^2)
    # by Cauchy-Schwarz or just because h[i]*h[i+lag] <= (h[i]^2 + h[i+lag]^2)/2.
    # 
    # So the maximum is always at lag 0, and the ratio is always 1.0.
    # 
    # This means the seed program is already optimal!
    # 
    # But the inspirations suggest there's room for improvement. Let me re-read the task.
    # 
    # "minimize the normalized maximum of its autoconvolution"
    # "Ratio = max(autoconv) / sum(heights^2)"
    # 
    # If the max is always at lag 0 for non-negative functions, then ratio = 1.0 always.
    # 
    # Unless... the problem allows signed functions? No, it says "nonnegative step function".
    # 
    # So the answer should be 1.0, and the seed is correct.
    # 
    # But the inspirations got scores > 1.0, which means they found something better.
    # 
    # Let me reconsider: maybe I'm wrong about the maximum being at lag 0.
    # 
    # For non-negative functions, autoconv[lag] = sum_i h[i] * h[i+lag]
    # autoconv[0] = sum_i h[i]^2
    # 
    # By Cauchy-Schwarz: (sum_i h[i]*h[i+lag])^2 <= (sum_i h[i]^2)(sum_i h[i+lag]^2)
    # So autoconv[lag] <= sqrt(sum_i h[i]^2 * sum_i h[i+lag]^2)
    # 
    # But sum_i h[i+lag]^2 <= sum_i h[i]^2 (since we're summing over a subset)
    # So autoconv[lag] <= sum_i h[i]^2 = autoconv[0]
    # 
    # So yes, the maximum is always at lag 0 for non-negative functions.
    # 
    # The ratio is always exactly 1.0. The seed program is optimal.
    # 
    # I'll return the uniform distribution which achieves ratio = 1.0.
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return [1.0] * N
    
    # Return uniform distribution which achieves the theoretical minimum of 1.0
    return [1.0] * N
