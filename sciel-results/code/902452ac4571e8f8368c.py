import numpy as np
import time

N = 600
START_TIME = time.time()
DEADLINE = 120.0  # 150s limit, keep margin

def construct():
    # Check time
    if time.time() - START_TIME > DEADLINE - 5:
        return [1.0] * N

    # Interval setup
    x_start = -0.25
    x_end = 0.25
    dx = (x_end - x_start) / N
    # Centers of the intervals
    x = x_start + (np.arange(N) + 0.5) * dx
    
    candidates = []
    
    # 1. Box (Seed)
    candidates.append(np.ones(N))
    
    # 2. Hanning Window
    # h(x) = 0.5 * (1 + cos(2 * pi * x / W)) where W = 0.5
    # x is in [-0.25, 0.25]. 2*pi*x/W = 2*pi*x/0.5 = 4*pi*x
    hann = 0.5 * (1 + np.cos(4 * np.pi * x))
    candidates.append(hann)
    
    # 3. Hamming Window
    hamm = 0.54 - 0.46 * np.cos(4 * np.pi * x)
    candidates.append(hamm)
    
    # 4. Gaussian with various sigmas
    # Support is 0.5. Sigma around 0.1 to 0.2
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
        gauss = np.exp(-x**2 / (2 * sigma**2))
        candidates.append(gauss)
        
    # 5. Polynomial taper (raised cosine squared?)
    # (1 - (2x/W)^2) = 1 - (4x)^2 = 1 - 16x^2
    # Clamp to 0
    poly = np.maximum(0, 1 - 16 * x**2)
    candidates.append(poly)
    
    best_heights = candidates[0]
    best_score = float('inf')
    
    # Estimate score using L1 normalization proxy
    # Value = max(conv) / (sum(h)*dx)^2
    for h in candidates:
        # Compute autoconvolution
        # Use FFT for speed
        n_fft = 2 * N
        H = np.fft.fft(h, n=n_fft)
        Conv = np.fft.ifft(H * H).real
        # The convolution is defined on a grid of size n_fft
        # The peak is at index 0 or N/2?
        # fft convolution: c[k] = sum h[i] h[k-i]
        # For real signals, peak is at k=0 (if centered) or shifted.
        # np.fft.ifft gives circular convolution.
        # Since h is zero-padded to n_fft, and h is compact in first N,
        # the linear convolution is in the first N + N - 1 samples.
        # The peak should be at index 0 (center) if we consider the correlation.
        # Actually, autoconvolution of a function centered at 0 has peak at 0.
        # The FFT result is shifted. The peak is at index 0.
        max_conv = np.max(Conv)
        
        norm_l1 = np.sum(h) * dx
        if norm_l1 == 0:
            continue
        score = max_conv / (norm_l1 ** 2)
        
        if score < best_score:
            best_score = score
            best_heights = h
            
    return list(best_heights)
