import numpy as np
import time

N = 100

def construct():
    """
    Construct step heights minimizing max overlap with shifted 1-h.
    
    Mathematical Analysis:
    Let h be the vector of step heights. We want to minimize max_k O_k,
    where O_k = sum_i h_i (1 - h_{(i+k) % N}).
    
    O_k = sum_i h_i - sum_i h_i h_{(i+k) % N}
    O_k = N/2 - C_k
    
    where C_k is the circular autocorrelation of h.
    
    We know that sum_k C_k = (sum_i h_i)^2 = (N/2)^2 = N^2/4.
    Therefore, the average value of C_k is N/4.
    This implies min_k C_k <= N/4.
    Consequently, max_k O_k = N/2 - min_k C_k >= N/2 - N/4 = N/4.
    
    The lower bound N/4 is achieved if and only if C_k is constant for all k,
    i.e., C_k = N/4 for all k.
    
    The uniform sequence h_i = 0.5 for all i satisfies:
    C_k = sum_i 0.5 * 0.5 = N * 0.25 = N/4.
    
    Thus, the uniform sequence achieves the theoretical minimum maximum overlap.
    Any other sequence will have variance in C_k, leading to some C_k < N/4,
    which results in some O_k > N/4.
    
    Therefore, the optimal solution is h_i = 0.5 for all i.
    """
    return np.full(N, 0.5)
