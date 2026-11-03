"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct step heights h[0..N-1] in [0,1] with sum = N/2
    to minimize the maximum overlap of h with shifted (1-h).
    
    The overlap for shift k is:
      O_k = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    
    We can achieve zero overlap by using a non-decreasing sequence
    consisting of zeros, a single fractional value (if needed), and ones.
    Specifically, let M = floor(N/2).
    Let c = N/2 - M.
    Let K = N - 1 - M.
    The sequence is:
      [0] * K + [c] + [1] * M
    
    This sequence is non-decreasing.
    For any i, if h[i] = 0, the term is 0.
    If h[i] = 1, then i is in the block of 1s, so i+k is also in the block of 1s (or out of bounds),
    meaning h[i+k] = 1, so 1 - h[i+k] = 0.
    If h[i] = c, then i is the single fractional element. i+k > i, so h[i+k] = 1,
    meaning 1 - h[i+k] = 0.
    Thus, all overlaps are 0.
    """
    target_sum = N / 2.0
    
    # Number of 1s
    M = int(np.floor(target_sum))
    # Fractional part
    c = target_sum - M
    # Number of 0s
    K = N - 1 - M
    
    # Construct the array
    h = np.zeros(N)
    if K > 0:
        h[:K] = 0.0
    h[K] = c
    if M > 0:
        h[K+1:] = 1.0
        
    return h

# EVOLVE-BLOCK-END
