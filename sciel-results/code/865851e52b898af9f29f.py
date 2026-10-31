import time
import numpy as np

N = 600

def compute_ratio(heights):
    h = np.asarray(heights, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s <= 0:
        return 1e10
    # Linear autoconvolution via FFT
    fft_size = 2 * N
    F = np.fft.rfft(h, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    max_conv = conv.max()
    return max_conv / (s * s)

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Baseline: box function
    box = np.ones(N)
    best = box.copy()
    best_score = compute_ratio(box)
    
    # Parameterize as cosine series: h_i = a0 + sum_{k=1}^{M} a_k cos(2*pi*k*i/N)
    # Symmetry: this is automatically symmetric about i=0 (and i=N/2)
    # We'll try M up to some order
    
    i = np.arange(N)
    M_max = 10
    
    # Initial: just a0
    for M in range(1, M_max + 1):
        if time.time() > deadline:
            break
        
        # Build cosine basis
        cos_basis = np.column_stack([np.cos(2 * np.pi * k * i / N) for k in range(1, M + 1)])
        
        # Optimize coefficients a0, a1, ..., aM
        # Use a simple iterative Remez-like approach or just try a few perturbations
        
        # Start from box (a0=1, others 0)
        a0 = 1.0
        a_rest = np.zeros(M)
        
        # Try small perturbations in each cosine direction
        for k in range(M):
            if time.time() > deadline:
                break
            
            for amp in [0.01, 0.05, 0.1, 0.2, 0.3]:
                for sign in [1, -1]:
                    a_test = np.zeros(M)
                    a_test[k] = sign * amp
                    h_test = a0 + cos_basis @ a_test
                    if h_test.min() < 0:
                        continue
                    score = compute_ratio(h_test)
                    if score < best_score:
                        best_score = score
                        best = h_test.copy()
        
        # Also try adjusting a0
        for a0_test in [0.5, 0.8, 1.2, 1.5, 2.0]:
            if time.time() > deadline:
                break
            h_test = a0_test + cos_basis @ a_rest
            if h_test.min() < 0:
                continue
            score = compute_ratio(h_test)
            if score < best_score:
                best_score = score
                best = h_test.copy()
    
    # Also try a few structured functions
    # Gaussian-like
    for sigma in [0.1, 0.2, 0.3, 0.5]:
        if time.time() > deadline:
            break
        center = N / 2
        h_gauss = np.exp(-0.5 * ((i - center) / (sigma * N)) ** 2)
        h_gauss = h_gauss / h_gauss.max()
        score = compute_ratio(h_gauss)
        if score < best_score:
            best_score = score
            best = h_gauss.copy()
    
    # Window functions
    for alpha in [0.1, 0.3, 0.5]:
        if time.time() > deadline:
            break
        t = i / N  # 0 to 1
        h_win = np.cos(np.pi * alpha * (2 * t - 1)) ** 2
        score = compute_ratio(h_win)
        if score < best_score:
            best_score = score
            best = h_win.copy()
    
    if time.time() > deadline:
        return np.ones(N).tolist()
    
    if best_score < compute_ratio(np.ones(N)):
        return best.tolist()
    else:
        return np.ones(N).tolist()
