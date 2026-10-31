"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve


def _score(signs):
    """Compute the C3-prime value for a binary sign sequence."""
    s = signs.astype(np.int32)
    integral = np.sum(s)
    if integral == 0:
        return np.inf
    denom = integral ** 2
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return float(max_ac) / denom


def construct():
    # The global minimum for the score max(|ac|) / (sum(s))^2 is achieved
    # when the sequence is constant (all 1s or all -1s).
    # For all 1s:
    # sum(s) = N
    # ac[0] = N, ac[k] = N-k for k>0.
    # max(|ac|) = N.
    # score = N / N^2 = 1/N.
    # Any deviation reduces the sum (denominator) and potentially increases
    # or maintains the max autoconvolution (numerator), thus increasing the score.
    
    signs = np.ones(N, dtype=np.int32)
    
    return signs
# EVOLVE-BLOCK-END
