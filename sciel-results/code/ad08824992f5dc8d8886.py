import numpy as np
import time

N = 600
DEADLINE = 140.0  # safety margin before 150s kill

def _autoconv_max(f):
    """Compute max of autoconvolution normalized by (integral(f))^2."""
    # f is a 1D array of step heights
    # Autoconvolution via FFT
    n = len(f)
    # Zero-pad to avoid circular convolution
    f_padded = np.zeros(2 * n - 1)
    f_padded[:n] = f
    conv = np.fft.ifft(np.fft.fft(f_padded) ** 2).real
    # The integral of f is sum(f) * (interval_width), but since we normalize
    # by (integral)^2, and the max is also scaled, the interval_width cancels
    integral = np.sum(f)
    if integral == 0:
        return 0.0
    return np.max(conv) / (integral ** 2)

def _coordinate_descent(f, max_iter=5000, tol=1e-8):
    """Coordinate descent to minimize autoconv_max."""
    f = f.copy()
    n = len(f)
    best_val = _autoconv_max(f)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            old_val = f[i]
            # Try a few candidate values for f[i]
            # We can try: 0, current*0.5, current*1.5, and some random values
            candidates = [0.0, old_val * 0.5, old_val * 1.5]
            if old_val > 0:
                candidates.append(old_val * 0.8)
                candidates.append(old_val * 1.2)
            
            best_cand = old_val
            best_cand_val = best_val
            
            for c in candidates:
                f[i] = c
                val = _autoconv_max(f)
                if val < best_cand_val:
                    best_cand_val = val
                    best_cand = c
            
            f[i] = best_cand
            if best_cand != old_val:
                improved = True
                best_val = best_cand_val
        
        if not improved:
            break
    
    return f, best_val

def construct():
    """
    Returns nonnegative step function heights minimizing the normalized
    maximum of autoconvolution.
    """
    start_time = time.time()
    
    # Start with uniform
    f_uniform = np.ones(N)
    val_uniform = _autoconv_max(f_uniform)
    
    best_f = f_uniform.copy()
    best_val = val_uniform
    
    # Try coordinate descent from uniform
    if time.time() < DEADLINE:
        f_cd, val_cd = _coordinate_descent(f_uniform, max_iter=100)
        if val_cd < best_val:
            best_f = f_cd
            best_val = val_cd
    
    # Try some shaped initializations
    if time.time() < DEADLINE:
        # Gaussian-like
        x = np.linspace(-1, 1, N)
        f_gauss = np.exp(-2 * x ** 2)
        f_gauss = f_gauss / np.sum(f_gauss) * N  # normalize to same integral as uniform
        f_cd, val_cd = _coordinate_descent(f_gauss, max_iter=50)
        if val_cd < best_val:
            best_f = f_cd
            best_val = val_cd
    
    if time.time() < DEADLINE:
        # Triangle
        f_tri = 1.0 - np.abs(np.linspace(-1, 1, N))
        f_tri = f_tri / np.sum(f_tri) * N
        f_cd, val_cd = _coordinate_descent(f_tri, max_iter=50)
        if val_cd < best_val:
            best_f = f_cd
            best_val = val_cd
    
    # Return as list of floats
    return [float(x) for x in best_f]
