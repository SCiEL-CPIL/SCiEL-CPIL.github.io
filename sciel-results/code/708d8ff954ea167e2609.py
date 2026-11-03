"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Theoretical analysis:
    Let C(k) = sum_i h[i] * (1 - h[(i+k)%N]).
    Summing C(k) over all k gives N^2/4.
    Thus, max_k C(k) >= N/4.
    The uniform vector h[i] = 0.5 yields C(k) = N/4 for all k.
    Therefore, the uniform vector is the unique optimal solution (up to permutation, 
    but uniform is the only one that is shift-invariant and achieves the bound for all shifts simultaneously in a simple way).
    
    We return the uniform vector directly as it is the global optimum.
    """
    # The optimal solution is h_i = 0.5 for all i
    # This ensures sum(h) = N * 0.5 = N/2
    # And max overlap is exactly N/4, which is the theoretical lower bound.
    
    h = np.full(N, 0.5)
    
    return h

# EVOLVE-BLOCK-END
