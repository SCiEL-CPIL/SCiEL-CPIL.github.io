"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 115  # seconds, safety margin before 150s kill

def construct():
    """
    The problem is to minimize the maximum overlap of h with shifted 1-h.
    Overlap at shift k: O_k = sum_i h_i (1 - h_{i+k}) = sum(h) - sum(h_i h_{i+k}).
    Given sum(h) = N/2, minimizing max(O_k) is equivalent to maximizing min(C_k),
    where C_k is the circular autocorrelation.
    
    The sum of all C_k is (sum(h))^2 = (N/2)^2 = N^2/4.
    The average C_k is N/4.
    Therefore, min(C_k) <= N/4.
    This implies max(O_k) >= N/2 - N/4 = N/4.
    
    The constant sequence h_i = 0.5 has C_k = N/4 for all k.
    Thus, max(O_k) = N/4.
    This is the global minimum.
    
    The optimal solution is the constant sequence 0.5.
    """
    h = np.full(N, 0.5)
    
    # Ensure exact sum constraint (0.5 is exact, but good practice)
    # sum(h) = N * 0.5 = 50.0 exactly for N=100
    # No correction needed.
    
    return h
# EVOLVE-BLOCK-END
