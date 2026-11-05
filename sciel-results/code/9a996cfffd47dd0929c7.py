"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import windows

def construct():
    """
    Construct step heights that minimize the autocorrelation metric.
    
    The metric is: value = abs(2 * N * max(convolve(h, h)) / sum(h)^2)
    
    We explore chirp signals (quadratic phase) multiplied by windows to flatten
    the convolution profile.
    """
    best_h = np.ones(N)
    best_val = 2.0 # Seed value
    
    # Helper to evaluate a candidate
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-10:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / s**2)
        return val

    t = np.arange(N)
    n_centered = t - (N-1)/2.0
    
    # 1. Try various chirp rates with Gaussian windows
    # h[n] = cos(pi * mu * n^2 / N) * exp(-n^2 / (2*sigma^2))
    chirp_rates = np.linspace(0.1, 5.0, 20) # mu
    sigmas = np.linspace(0.1, 0.8, 10) # relative sigma (fraction of N)
    
    for mu in chirp_rates:
        for sigma in sigmas:
            window = np.exp(-n_centered**2 / (2 * (sigma * N)**2))
            phase = np.pi * mu * n_centered**2 / N
            h = np.cos(phase) * window
            val = evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h
            
            # Also try sine
            h_sin = np.sin(phase) * window
            val_sin = evaluate(h_sin)
            if val_sin < best_val:
                best_val = val_sin
                best_h = h_sin

    # 2. Try Tukey windows with chirps
    for mu in chirp_rates:
        for alpha in np.linspace(0.1, 1.0, 10): # taper fraction
            window = windows.tukey(N, alpha=alpha)
            phase = np.pi * mu * n_centered**2 / N
            h = np.cos(phase) * window
            val = evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h
            
            h_sin = np.sin(phase) * window
            val_sin = evaluate(h_sin)
            if val_sin < best_val:
                best_val = val_sin
                best_h = h_sin

    # 3. Try Zadoff-Chu like sequences (complex exponential real part)
    # z[n] = exp(-j * pi * root * n * (n+1) / N)
    # Real part is cos(pi * root * n * (n+1) / N)
    roots = np.arange(1, 20)
    for root in roots:
        for sigma in sigmas:
            window = np.exp(-n_centered**2 / (2 * (sigma * N)**2))
            phase = -np.pi * root * t * (t+1) / N
            h = np.cos(phase) * window
            val = evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h
            
            h_sin = np.sin(phase) * window
            val_sin = evaluate(h_sin)
            if val_sin < best_val:
                best_val = val_sin
                best_h = h_sin

    return best_h
# EVOLVE-BLOCK-END
