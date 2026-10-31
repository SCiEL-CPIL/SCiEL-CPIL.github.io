import numpy as np

N = 100

def construct():
    """
    Returns a sequence of N=100 step heights in [0, 1] whose sum is exactly N/2 = 50.
    The constant 0.5 sequence is the provable global optimum for minimizing the
    maximum overlap of h with shifted 1-h.
    
    Proof:
    Let C_k be the circular autocorrelation. The overlap at shift k is O_k = N/2 - C_k.
    We want to minimize max(O_k), which is equivalent to maximizing min(C_k).
    The average of C_k over all shifts is (sum(h))^2 / N = (N/2)^2 / N = N/4.
    Since min(C_k) <= average(C_k) = N/4, we have min(C_k) <= N/4.
    This implies max(O_k) = N/2 - min(C_k) >= N/2 - N/4 = N/4.
    The constant sequence h_i = 0.5 yields C_k = N/4 for all k, so min(C_k) = N/4
    and max(O_k) = N/4.
    Thus, it is the unique global optimum (up to permutation, but constant is simplest).
    """
    return np.full(N, 0.5)
