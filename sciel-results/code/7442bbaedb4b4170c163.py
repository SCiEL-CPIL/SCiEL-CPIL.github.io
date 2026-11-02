<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 115.0

def _autocorr(h):
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real

def _max_overlap(h):
    r = _autocorr(h)
    return HALF - float(np.min(r[1:]))
=======
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 115.0

def _autocorr(h):
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real

def _max_overlap(h):
    r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _max_overlap_fast(h, r_cache=None):
    """Compute max overlap, optionally using cached autocorrelation."""
    if r_cache is not None:
        r = r_cache
    else:
        r = _autocorr(h)
    return HALF - float(np.min(r[1:]))

def _update_autocorr_after_swap(r, h, i, j):
    """
    Update autocorrelation after swapping h[i] and h[j].
    Only shifts k where |i-j| mod N != k are affected.
    Actually, all shifts are potentially affected, but we can compute delta efficiently.
    
    For a swap of positions i and j, the change in autocorrelation at shift k is:
    delta_r[k] = (h[j]-h[i]) * (h[(i+k)%N] - h[(j+k)%N]) + (h[i]-h[j]) * (h[(i-k)%N] - h[(j-k)%N])
    Wait, this isn't quite right. Let me think more carefully.
    
    r[k] = sum_m h[m] * h[(m+k) % N]
    
    After swapping h[i] <-> h[j], let a = h[i], b = h[j].
    New h'[i] = b, h'[j] = a, others unchanged.
    
    r'[k] = r[k] - a*h[(i+k)%N] - b*h[(j+k)%N] + a*h[(j-k)%N] + b*h[(i-k)%N]
           Wait, this is getting complicated. Let me just recompute.
    """
    # For simplicity and correctness, just recompute. N=100 is small.
    return _autocorr(h)
>>>>>>> REPLACE

