<<<<<<< SEARCH
import numpy as np
from scipy import optimize

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = np.convolve(h, h, 'full')
    total = sum(h)
    if total == 0:
        return 0.0
    return max(conv) / (total * total)
=======
import numpy as np
from scipy import optimize

N = 600

def _fft_convolve(a, b):
    """Fast convolution using FFT."""
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    fc = fa * fb
    c = np.fft.irfft(fc, nfft)
    return c[:n]

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h, h)
    total = sum(h)
    if total == 0:
        return 0.0
    return max(conv) / (total * total)
>>>>>>> REPLACE

