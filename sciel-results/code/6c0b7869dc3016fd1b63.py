import numpy as np
import time

N = 400

def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes."""
    # Use FFT for efficient autoconvolution
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    center = N - 1
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    candidates = []
    
    # Try narrow boxcars (widths 1, 2, 3, 5, 10)
    for width in [1, 2, 3, 5, 10, 20, 50, 100, 200, 400]:
        if time.time() > deadline - 10:
            break
        f = np.zeros(N)
        # Center the boxcar
        start_idx = (N - width) // 2
        end_idx = start_idx + width
        f[start_idx:end_idx] = 1.0
        norm = np.linalg.norm(f)
        if norm == 0:
            continue
        f = f / norm
        val = _get_max_sidelobe(f)
        candidates.append((val, f))
        
    # Try a narrow Gaussian
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        if time.time() > deadline - 10:
            break
        x = np.linspace(-N/2, N/2, N)
        f = np.exp(-x**2 / (2 * sigma**2))
        norm = np.linalg.norm(f)
        if norm == 0:
            continue
        f = f / norm
        val = _get_max_sidelobe(f)
        candidates.append((val, f))

    if not candidates:
        # Fallback to Delta
        f = np.zeros(N)
        f[N//2] = 1.0
        return f

    # Find the best candidate
    best_val, best_f = min(candidates, key=lambda x: x[0])
    
    return best_f
