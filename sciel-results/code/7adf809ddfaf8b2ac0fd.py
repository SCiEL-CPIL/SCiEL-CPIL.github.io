import numpy as np
from scipy import signal
import time

N = 600

def compute_score(heights):
    """Compute normalized max of autoconvolution for step function heights."""
    # Interval width
    total_width = 0.5  # from -1/4 to 1/4
    w = total_width / N
    
    # Create a fine grid to represent the step function
    # Use enough points to resolve the convolution
    n_fine = 4096  # power of 2 for FFT efficiency
    x = np.linspace(-total_width/2, total_width/2, n_fine, endpoint=False)
    
    # Create the step function on the fine grid
    # Each interval i covers [x_start + i*w, x_start + (i+1)*w)
    x_start = -total_width / 2
    f = np.zeros(n_fine)
    for i in range(N):
        interval_start = x_start + i * w
        interval_end = x_start + (i + 1) * w
        mask = (x >= interval_start) & (x < interval_end)
        f[mask] = heights[i]
    
    # Compute autoconvolution via FFT
    f_fft = np.fft.fft(f, n=2*n_fine)
    conv_fft = f_fft * np.conj(f_fft)
    conv = np.fft.ifft(conv_fft).real
    conv = conv[:n_fine] * (total_width / n_fine)  # scale by dx
    
    max_conv = np.max(conv)
    
    # L2 norm squared
    l2_sq = np.sum(f**2) * (total_width / n_fine)
    
    if l2_sq < 1e-15:
        return float('inf')
    
    return max_conv / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with the box (score 1.0)
    best_heights = [1.0] * N
    best_score = compute_score(best_heights)
    
    # Try various patterns
    patterns_to_try = []
    
    # Comb patterns with different periods
    for period in range(2, 20):
        h = [1.0 if i % period == 0 else 0.0 for i in range(N)]
        patterns_to_try.append(h)
    
    # Block patterns
    for block_size in range(1, 30):
        for gap_size in range(1, 30):
            if block_size + gap_size > N:
                break
            h = [0.0] * N
            for i in range(0, N, block_size + gap_size):
                for j in range(block_size):
                    if i + j < N:
                        h[i + j] = 1.0
            patterns_to_try.append(h)
    
    # Random patterns
    np.random.seed(42)
    for _ in range(50):
        density = np.random.uniform(0.1, 0.8)
        h = (np.random.rand(N) < density).astype(float)
        patterns_to_try.append(h.tolist())
    
    # Evaluate patterns
    for h in patterns_to_try:
        if time.time() > deadline - 10:
            break
        score = compute_score(h)
        if score < best_score:
            best_score = score
            best_heights = h
    
    # Local improvement on best
    if time.time() < deadline - 30:
        current = list(best_heights)
        for iteration in range(1000):
            if time.time() > deadline - 5:
                break
            # Randomly flip a few positions
            n_flips = np.random.randint(1, 6)
            indices = np.random.choice(N, n_flips, replace=False)
            candidate = list(current)
            for idx in indices:
                candidate[idx] = 1.0 - candidate[idx]  # toggle between 0 and 1
            score = compute_score(candidate)
            if score < best_score:
                best_score = score
                best_heights = candidate
                current = candidate
    
    return best_heights
