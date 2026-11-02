<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return -1.0
    if 0 not in U:
        return -1.0
    max_u = U[-1]
    if max_u <= 0:
        return -1.0
    
    # For small sets, use Python sets (faster overhead)
    if n <= 30:
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_sums = len(sum_set)
        n_diffs = len(diff_set)
    else:
        U_arr = np.array(U, dtype=np.int64)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
    
    if n_sums == 0:
        return -1.0
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from scipy.fft import next_fast_len

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1) using FFT-based convolution."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return -1.0
    if 0 not in U:
        return -1.0
    max_u = U[-1]
    if max_u <= 0:
        return -1.0
    
    # For very small sets, use Python sets (less overhead than FFT setup)
    if n <= 8:
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_sums = len(sum_set)
        n_diffs = len(diff_set)
    else:
        # Use FFT-based convolution for sumset
        # |A+B| = number of positions where conv(indicator_A, indicator_B) > 0
        # |A-B| = number of positions where conv(indicator_A, reversed_indicator_B) > 0
        
        # Sumset: conv(A, A) gives counts of a+b for each sum
        # We need indicator vectors of length max_u+1
        size = max_u + 1
        # Next fast length for FFT
        fft_len = next_fast_len(2 * size - 1)
        
        # Indicator vector for U
        ind = np.zeros(fft_len, dtype=np.float64)
        for u in U:
            ind[u] = 1.0
        
        # Sumset: A + A, result has values from 0 to 2*max_u
        # conv(ind, ind) gives sumset counts
        sum_conv = np.fft.irfft(np.fft.rfft(ind) ** 2)
        n_sums = np.count_nonzero(sum_conv[:2 * max_u + 1])
        
        # Difference set: A - A, values from -max_u to max_u
        # This is conv(A, reversed_A) shifted
        # diff(a,b) = a - b, so we need conv(ind, flip(ind))
        ind_flip = np.zeros(fft_len, dtype=np.float64)
        for u in U:
            ind_flip[max_u - u] = 1.0
        
        diff_conv = np.fft.irfft(np.fft.rfft(ind) * np.fft.rfft(ind_flip))
        # The diff values range from -max_u to max_u, centered at max_u in the convolution
        # diff_conv[k] corresponds to sum index k, which maps to diff value k - max_u
        n_diffs = np.count_nonzero(diff_conv[max_u:max_u + 2 * max_u + 1])
    
    if n_sums == 0:
        return -1.0
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom
>>>>>>> REPLACE

