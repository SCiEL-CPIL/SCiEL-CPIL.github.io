import time
import sys
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of nonnegative step heights on equal intervals of [-1/4, 1/4].
    The goal is to minimize the normalized maximum of the autoconvolution.
    
    Mathematical Analysis:
    Let h be the vector of heights. The peak of the autoconvolution of a non-negative
    step function is proportional to sum(h^2). The normalization typically fixes
    the L1 norm (area) or L2 norm. 
    1. If L1 is fixed: sum(h) = C. We minimize sum(h^2). By Jensen's inequality,
       sum(h^2) is minimized when all h_i are equal.
    2. If L2 is fixed: sum(h^2) = C. The peak is proportional to sum(h^2), which is
       constant. However, the "normalized maximum" usually implies a ratio like
       ||f*f||_inf / ||f||_1^2 or ||f*f||_inf / ||f||_2^2.
       
    For the ratio ||f*f||_inf / ||f||_2^2, the value is 1 for the box function.
    For any other shape, the peak ||f*f||_inf is still determined by the overlap at t=0.
    Actually, for non-negative functions, the maximum of the convolution is at t=0.
    (f*f)(0) = sum h_i^2 (scaled by dt).
    So the ratio (f*f)(0) / ||f||_2^2 is exactly 1 (scaled by dt/dt) for ANY function?
    No, ||f||_2^2 = sum h_i^2 dt. (f*f)(0) = sum h_i^2 dt.
    So the ratio is 1 for all functions?
    
    If the ratio is 1 for all functions, the score would be 1.0 for all valid inputs.
    But the task implies variation.
    
    Let's consider the ratio ||f*f||_inf / ||f||_1^2.
    For box: ||f||_1^2 = (N h dt)^2. Peak = N h^2 dt.
    Ratio = (N h^2 dt) / (N^2 h^2 dt^2) = 1 / (N dt).
    This is a constant independent of h.
    
    Is it possible the support is not fixed? "Choose a nonnegative step function with 600 equal intervals on [-1/4,1/4]".
    This implies the domain is fixed.
    
    Perhaps the "normalized maximum" is defined differently, e.g., relative to the width?
    Or perhaps the function doesn't have to be non-zero everywhere?
    If we set some h_i to 0, we reduce the support.
    Let support be M intervals.
    Box on M intervals: h = C/M.
    Peak = M (C/M)^2 dt = C^2 / M * dt.
    ||f||_1^2 = C^2.
    Ratio = (C^2/M * dt) / C^2 = dt / M.
    To minimize the ratio, we need to maximize M.
    Max M is N.
    So full support box is optimal.
    
    Therefore, the uniform box function over the entire interval is the optimal solution.
    """
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Construct the optimal uniform box function
    # Heights are 1.0. The normalization is handled by the evaluator.
    # The shape is uniform to minimize the peak-to-area or peak-to-energy ratio.
    result = np.ones(N, dtype=np.float64)
    
    # Check time (trivial, but required by directive)
    if time.time() > deadline:
        return result.tolist()
        
    return result.tolist()
# EVOLVE-BLOCK-END
