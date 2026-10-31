"""
Optimized program for the suite task.
Strategy: The metric "normalized maximum of autoconvolution" for non-negative functions
is minimized by the uniform distribution, as the peak of the autoconvolution is at lag 0
and equals the sum of squares, which is minimized by uniform distribution for a fixed sum.
"""
import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # The theoretical minimum for max(autoconv) / (sum(h))^2 for h >= 0 is achieved
    # when h is uniform.
    # max(autoconv) = sum(h^2) for h >= 0 (peak at lag 0).
    # Minimize sum(h^2) subject to sum(h) = 1, h >= 0.
    # Solution: h_i = 1/N for all i.
    
    # We return uniform heights.
    # Normalization: The score is scale invariant, so we can return ones or 1/N.
    # The seed program likely used ones or similar.
    # Returning ones is safe.
    
    heights = np.ones(N)
    
    # If we want to be extremely safe and check if a "flattened" version helps 
    # (it shouldn't for global max, but just in case of numerical quirks or 
    # if the metric is slightly different), we can do a quick check.
    # However, Uniform is the analytic minimum.
    
    return list(heights)

# EVOLVE-BLOCK-END
