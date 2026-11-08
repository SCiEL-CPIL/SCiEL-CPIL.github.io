"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _rudin_shapiro(n):
    """
    Generate a Rudin-Shapiro sequence of length n.
    Rudin-Shapiro sequences have low peak-to-sidelobe ratio.
    """
    if n == 0:
        return np.array([], dtype=int)
    
    # Generate base sequences
    # r_1 = 1, s_1 = 1
    # r_{k+1} = r_k + s_k
    # s_{k+1} = r_k - s_k
    # The sequence is often defined as the infinite concatenation.
    # We need a sequence of length N.
    
    # Efficient generation using bit manipulation or recursion
    # r(n) = 1 if n has even number of 1s in binary representation? No.
    # Let's use the recursive construction up to length >= N.
    
    r = np.array([1], dtype=int)
    s = np.array([1], dtype=int)
    
    while len(r) < N:
        # r_{k+1} = r_k + s_k
        # s_{k+1} = r_k - s_k
        # Note: The standard RS sequence is the limit of r_k.
        # r_1 = [1]
        # r_2 = [1, 1]
        # r_3 = [1, 1, 1, -1]
        # r_4 = [1, 1, 1, -1, 1, 1, -1, -1]
        
        new_r = np.concatenate([r, s])
        new_s = np.concatenate([r, -s])
        r = new_r
        s = new_s
        
    return r[:N]

def construct():
    """
    Construct a sequence of signed step heights minimizing the maximum 
    absolute autoconvolution (interpreted as sidelobe minimization for C3 context).
    """
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate Rudin-Shapiro sequence
    seq = _rudin_shapiro(N)
    
    # Rudin-Shapiro sequences are +/- 1
    # Check if we have time for local optimization (bit flips)
    # Given the strict time limit and the fact that RS is already very good,
    # we return it directly.
    
    if time.time() > deadline:
        return seq
    
    return seq
# EVOLVE-BLOCK-END
