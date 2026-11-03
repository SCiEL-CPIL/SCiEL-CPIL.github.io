"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Theoretical analysis:
    - For any h with sum N/2 and 0 <= h_i <= 1:
      C(k) = sum_i h[i] * (1 - h[i+k]) for circular shift k
      Sum over all k: sum_k C(k) = (N/2) * (N - N/2) = N^2/4
      Average C(k) = N/4
      Therefore max_k C(k) >= N/4
    - Uniform h_i = 0.5 gives C(k) = N * 0.25 = N/4 for all k
    - So uniform achieves the lower bound and is optimal.
    
    Strategy:
    - Directly construct the uniform solution h_i = 0.5.
    - This is provably optimal: it achieves the information-theoretic lower bound.
    """
    # The optimal solution is the uniform distribution h_i = 0.5
    # This achieves C(k) = N/4 for all k, which is the minimum possible max overlap.
    h = np.full(N, 0.5)
    
    # Verify sum is exactly N/2
    total = np.sum(h)
    if total != N / 2.0:
        h = h * (N / 2.0) / total
    
    return h

# EVOLVE-BLOCK-END
