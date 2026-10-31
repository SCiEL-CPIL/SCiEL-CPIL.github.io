"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal signed step function.
    
    The C3-prime metric is max(|autoconv|) / (sum(signs))^2.
    For binary signs s_i in {-1, 1}, the autoconvolution at lag 0 is 
    sum(s_i^2) = N. Thus max(|autoconv|) >= N.
    The denominator is (sum(s_i))^2. To minimize the ratio, we must 
    maximize the denominator, which occurs when all signs are the same 
    (all 1s or all -1s), giving a sum of +/- N and denominator N^2.
    The minimum metric is N / N^2 = 1/N.
    
    Therefore, the constant sequence of 1s (or -1s) is the global optimum.
    """
    return np.ones(N, dtype=np.int32)
# EVOLVE-BLOCK-END
