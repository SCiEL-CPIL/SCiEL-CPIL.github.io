"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    
    # The mathematical optimum for the metric C3' = max(|A_k|) / (sum(x))^2
    # with x_i in {-1, 1} is the constant sequence.
    # For x = [1, 1, ..., 1]:
    # sum(x) = N
    # A_0 = N, A_k = N-k
    # max(|A_k|) = N
    # C3' = N / N^2 = 1/N
    # Any deviation reduces |sum(x)| or increases max(|A_k|) relative to sum^2,
    # resulting in a higher value.
    
    # Check if we have time (trivially yes, but compliant with directive)
    if time.time() - start_time > deadline - 0.1:
        return np.ones(N)
        
    return np.ones(N)
# EVOLVE-BLOCK-END
