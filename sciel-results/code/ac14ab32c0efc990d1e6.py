"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score_normalized(h):
    """
    Compute the objective: max|autoconv| / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    max_ac = np.max(np.abs(ac[:size]))
    return max_ac / (s * s)

def construct():
    # Approach: Maximize the sum of the sequence (S) to minimize the denominator's impact
    # on the ratio, assuming max|ac| scales with N for +/- 1 sequences.
    # The optimal +/- 1 sequence for minimizing max|ac|/S^2 is the one with largest S.
    # Largest S is N (all +1).
    # If all +1 is not allowed (e.g. must have both signs), use N-1 plus, 1 minus.
    
    # Try all +1
    h_all_plus = np.ones(N)
    val_plus = _score_normalized(h_all_plus)
    
    # Try N-1 plus, 1 minus (at index 0)
    h_one_minus = np.ones(N)
    h_one_minus[0] = -1
    val_one_minus = _score_normalized(h_one_minus)
    
    # Try N-2 plus, 2 minus
    h_two_minus = np.ones(N)
    h_two_minus[0] = -1
    h_two_minus[1] = -1
    val_two_minus = _score_normalized(h_two_minus)
    
    best_h = h_all_plus
    best_val = val_plus
    
    if val_one_minus < best_val:
        best_h = h_one_minus
        best_val = val_one_minus
        
    if val_two_minus < best_val:
        best_h = h_two_minus
        best_val = val_two_minus

    # If the problem implies "signed" means we must use both, and all+1 is invalid,
    # the one_minus case is the best.
    # However, usually "signed" just means the values can be negative.
    # We return the best found.
    
    return best_h

# EVOLVE-BLOCK-END
