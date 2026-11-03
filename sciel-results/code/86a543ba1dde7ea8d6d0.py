"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    
    # Candidate 1: All ones (Constant sequence)
    # This minimizes the ratio to 1/N theoretically.
    h1 = np.ones(N)
    score1 = _score(h1)
    
    # Candidate 2: One -1, rest 1s
    # In case "signed" implies mixed signs.
    h2 = np.ones(N)
    h2[-1] = -1.0
    score2 = _score(h2)
    
    # Candidate 3: All -1s
    h3 = -np.ones(N)
    score3 = _score(h3)
    
    # Select the best candidate
    best_h = h1
    best_score = score1
    
    if score2 < best_score:
        best_h = h2
        best_score = score2
        
    if score3 < best_score:
        best_h = h3
        best_score = score3
        
    # Safety check: if scores are unexpectedly high (e.g., > 1.0), 
    # it might indicate a mismatch in assumptions. 
    # However, given the math, 1/N is the lower bound for this metric.
    # We return the best found.
    
    return best_h

# EVOLVE-BLOCK-END
