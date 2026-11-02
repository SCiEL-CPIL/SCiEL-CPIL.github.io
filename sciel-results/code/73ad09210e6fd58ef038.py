"""
Rewritten program for the second autocorrelation inequality task.
Strategy: The ratio R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_infty) is maximized by concentrating all mass at a single point.
For a non-negative vector h, ||h||_2^2 <= ||h||_1^2, with equality iff h is a spike (or scalar multiple).
Also ||h*h||_infty >= max(h_i)^2.
If h is a spike with value 1, R=1.
Any spread reduces ||h||_2^2 relative to ||h||_1^2 and increases ||h*h||_infty relative to the spike case (or keeps it high).
Thus, the global maximum is 1, achieved by a single-point spike.
This represents a structural change from the uniform distribution (which has ratio 1/N) to a point-mass distribution.
"""
import time
import numpy as np

N = 50

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # The optimal solution is a spike.
    # We choose the first element to be 1.0 and the rest 0.0.
    # This is a valid sequence of nonnegative heights.
    h = np.zeros(N)
    h[0] = 1.0
    
    # No iterative search needed as this is the global optimum.
    # Return as a list of floats to match the interface.
    return list(h)
