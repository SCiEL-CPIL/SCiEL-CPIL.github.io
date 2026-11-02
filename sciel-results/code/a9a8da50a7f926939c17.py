import numpy as np
import time

N = 600

def compute_score(h):
    """
    Compute the normalized maximum autoconvolution score.
    Score = max_k S_k / S_0, where S_k = sum_i h_i h_{i+k}
    """
    H = np.sum(h)
    if H < 1e-12:
        return np.inf
    
    # Compute autoconvolution coefficients S_k using FFT for efficiency
    n_fft = 2 * N
    H_fft = np.fft.rfft(h, n=n_fft)
    S_fft = H_fft * np.conj(H_fft)
    S = np.fft.irfft(S_fft, n=n_fft)[:N]
    
    S_0 = S[0]
    if S_0 < 1e-12:
        return np.inf
    
    return np.max(S) / S_0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize with uniform distribution
    h = np.ones(N)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # The problem is to minimize max_k S_k / S_0.
    # For non-negative h, S_0 = sum(h^2).
    # For uniform h, all S_k are equal, so max(S)/S_0 = 1.
    # This is the Cauchy-Schwarz bound.
    # We can try to find a non-uniform h that achieves max(S)/S_0 < 1.
    
    # Try a few simple patterns
    patterns = [
        np.ones(N),
        np.ones(N) * np.linspace(0.5, 1.5, N),
        np.ones(N) * (1.0 + 0.5 * np.cos(2 * np.pi * np.arange(N) / N)),
        np.ones(N) * (1.0 + 0.5 * np.cos(4 * np.pi * np.arange(N) / N)),
        np.ones(N) * (1.0 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)),
    ]
    
    for p in patterns:
        if np.min(p) < 0:
            p = np.clip(p, 0, None)
        if np.sum(p) < 1e-12:
            continue
        s = compute_score(p)
        if s < best_score:
            best_score = s
            best_h = p.copy()
    
    return list(best_h)

construct()
