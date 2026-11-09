import time
import numpy as np
from scipy.fft import fft, ifft

N = 600
DEADLINE_FRACTION = 0.85  # safety margin

def _autoconv_max(h):
    """Compute max of autoconvolution of h using FFT."""
    n = len(h)
    # Pad to next power of 2 for efficient FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = fft(h, n=size)
    autoconv = ifft(H * H, n=size).real
    return np.max(autoconv)

def _score(h):
    """Compute the normalized maximum of autoconvolution.
    
    The score is max(autoconv(h)) / (sum(h))^2, which is scale-invariant.
    Wait, let me reconsider. The task says "minimize the normalized maximum of its autoconvolution".
    For a step function with heights h_i on intervals of width w=1/300, the autoconvolution at the peak
    relates to sum h_i^2. Let me think more carefully.
    
    Actually, for the uniform case h_i = 1/N, the autoconvolution max is proportional to sum(h_i^2).
    The "normalized" likely means we normalize so that sum(h) = 1 (or some fixed L1 norm),
    and then minimize max(autoconv).
    
    Given the seed returns h_i = 1/N (sum = 1), and scores 1.0, the score is likely:
    score = max(autoconv(h)) / C where C is some normalization constant.
    
    Let me just compute: for uniform h_i = 1/N on N intervals of width w = 1/300,
    the autoconvolution is a triangle. Its max is at the center.
    
    Actually, let me just use the formula that makes uniform = 1.0.
    """
    s = np.sum(h)
    if s == 0:
        return float('inf')
    # Normalize to L1 = 1
    h_norm = h / s
    max_ac = _autoconv_max(h_norm)
    # For uniform, max_ac = 1/N (approximately, for large N)
    # Let's figure out the normalization: uniform should give score 1.0
    # The raw autoconv max for uniform h_i = 1/N: 
    # autoconv[k] = sum_i h_i * h_{i+k} (discrete)
    # For uniform, this is a triangle with max = sum_i (1/N)^2 = N * 1/N^2 = 1/N
    # So score = max_ac * N? That gives 1.0 for uniform.
    return max_ac * N

def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    # Start with uniform distribution
    h = np.ones(N) / N
    best_h = h.copy()
    best_score = _score(best_h)
    
    # Precompute FFT setup
    size = 1
    while size < 2 * N - 1:
        size *= 2
    
    # Coordinate descent with random restarts
    rng = np.random.default_rng(42)
    
    # Try several strategies
    strategies = [
        'uniform',
        'bump',
        'gaussian',
        'triangle'
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
            
        if strategy == 'uniform':
            h = np.ones(N) / N
        elif strategy == 'bump':
            # Parabolic bump
            x = np.linspace(-1, 1, N)
            h = np.maximum(0, 1 - x**2)
            h /= np.sum(h)
        elif strategy == 'gaussian':
            x = np.linspace(-1, 1, N)
            h = np.exp(-2 * x**2)
            h /= np.sum(h)
        elif strategy == 'triangle':
            x = np.linspace(-1, 1, N)
            h = np.maximum(0, 1 - np.abs(x))
            h /= np.sum(h)
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Coordinate descent
        for iteration in range(5000):
            if time.time() > deadline:
                break
                
            # Pick a random coordinate
            i = rng.integers(0, N)
            
            # Try adjusting h[i]
            # Optimal h[i] given others: minimize sum_j h_i * h_{j} * h_{j+i} etc.
            # The contribution to autoconv max from h[i] is complex.
            # Instead, try small perturbations
            
            orig = h[i]
            h[i] = max(0, orig + rng.normal(0, 1.0 / N))
            
            # Renormalize
            s = np.sum(h)
            if s > 0:
                h /= s
            
            new_score = _score(h)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                h[i] = orig
                s = np.sum(h)
                h /= s
        
    # One more round of local refinement
    if time.time() < deadline:
        h = best_h.copy()
        current_score = best_score
        for iteration in range(2000):
            if time.time() > deadline:
                break
            i = rng.integers(0, N)
            orig = h[i]
            # Try a few perturbations
            best_local = orig
            best_local_score = current_score
            for delta in [0.1/N, 0.5/N, 1.0/N, -0.1/N, -0.5/N, -1.0/N]:
                h[i] = max(0, orig + delta)
                s = np.sum(h)
                if s > 0:
                    h /= s
                sc = _score(h)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local = h[i]
            h[i] = best_local
            s = np.sum(h)
            if s > 0:
                h /= s
            current_score = best_local_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
    
    return [float(x) for x in best_h]
