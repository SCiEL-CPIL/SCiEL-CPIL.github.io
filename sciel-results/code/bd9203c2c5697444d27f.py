import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    dx = (1.0/4.0 - (-1.0/4.0)) / N  # = 1/600
    
    def autoconv_max(heights):
        """Compute max of autoconvolution of step function."""
        h = np.array(heights, dtype=np.float64)
        # Autoconvolution via FFT for efficiency
        # The autoconvolution of a step function on N intervals:
        # (f*f)[k] = sum_i h[i] * h[i+k] * dx for lag k (in interval units)
        # Actually, for discrete step functions, the convolution is:
        # c[m] = sum_{i} h[i] * h[i+m] * dx  for m = -(N-1)..(N-1)
        # But we need to be careful about the geometry.
        
        # Using FFT: pad to 2N-1 length
        h_padded = np.zeros(2*N - 1, dtype=np.float64)
        h_padded[:N] = h
        # Linear convolution via FFT
        H = np.fft.fft(h_padded, n=4*N)
        conv = np.fft.ifft(H * np.conj(H)).real[:2*N-1]
        # Scale by dx
        conv *= dx
        return np.max(conv)
    
    def score(heights):
        """Compute the normalized max - assuming normalization by (integral)^2"""
        h = np.array(heights, dtype=np.float64)
        integral = np.sum(h) * dx
        if integral < 1e-15:
            return float('inf')
        return autoconv_max(heights) / (integral * integral)
    
    best_heights = None
    best_score = float('inf')
    
    # Candidate 1: Uniform (seed)
    h1 = [1.0] * N
    s1 = score(h1)
    best_heights = h1
    best_score = s1
    
    # Candidate 2: Cosine bump
    x = np.linspace(-1.0/4.0, 1.0/4.0, N, endpoint=False) + dx/2
    h2 = np.cos(np.pi * x / (1.0/4.0)) ** 2  # cos^2 bump, zero at edges
    h2 = np.maximum(h2, 0)
    # Normalize to have same max as uniform
    if np.max(h2) > 0:
        h2 = h2 / np.max(h2)
    s2 = score(h2)
    if s2 < best_score:
        best_score = s2
        best_heights = h2.tolist()
    
    # Candidate 3: Triangular (peak in middle)
    h3 = np.zeros(N)
    mid = N // 2
    for i in range(N):
        h3[i] = 1.0 - abs(i - mid) / mid
    h3 = np.maximum(h3, 0)
    s3 = score(h3)
    if s3 < best_score:
        best_score = s3
        best_heights = h3.tolist()
    
    # Candidate 4: Two lobes (zero in middle)
    h4 = np.zeros(N)
    quarter = N // 4
    h4[:quarter] = 1.0
    h4[3*quarter:] = 1.0
    s4 = score(h4)
    if s4 < best_score:
        best_score = s4
        best_heights = h4.tolist()
    
    # Candidate 5: Smooth two-lobed
    h5 = np.zeros(N)
    for i in range(N):
        t = i / N  # 0 to 1
        # Two Gaussians centered at 0.25 and 0.75
        g1 = np.exp(-((t - 0.25) ** 2) / (2 * 0.05 ** 2))
        g2 = np.exp(-((t - 0.75) ** 2) / (2 * 0.05 ** 2))
        h5[i] = max(g1, g2)
    if np.max(h5) > 0:
        h5 = h5 / np.max(h5)
    s5 = score(h5)
    if s5 < best_score:
        best_score = s5
        best_heights = h5.tolist()
    
    # Candidate 6: Uniform but with slight modulation
    h6 = np.ones(N)
    h6 = 1.0 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N)
    h6 = np.maximum(h6, 0)
    s6 = score(h6)
    if s6 < best_score:
        best_score = s6
        best_heights = h6.tolist()
    
    # Check deadline
    if time.time() > deadline:
        return best_heights
    
    # Try some random perturbations of the best
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        base = np.array(best_heights, dtype=np.float64)
        perturbed = base + 0.1 * np.random.randn(N)
        perturbed = np.maximum(perturbed, 0)
        sp = score(perturbed)
        if sp < best_score:
            best_score = sp
            best_heights = perturbed.tolist()
    
    return best_heights
# EVOLVE-BLOCK-END
