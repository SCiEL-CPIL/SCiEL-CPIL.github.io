"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlap(h):
    """Compute max over all shifts k of sum_i h[i] * (1 - h[(i+k) % N])."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        shifted = np.roll(h, -k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_cosine(f, A, phi):
    i = np.arange(N, dtype=np.float64)
    h = 0.5 + A * np.cos(2.0 * np.pi * f * i / N + phi)
    h = np.clip(h, 0.0, 1.0)
    # Adjust to make sum exactly N/2
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (N / 2.0) / current_sum
        h = np.clip(h, 0.0, 1.0)
        # Re-adjust after clipping
        current_sum = np.sum(h)
        if current_sum > 0:
            h = h * (N / 2.0) / current_sum
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap(best_h)
    
    # Search over frequency, amplitude, phase
    # Try a grid search over structured parameters
    freqs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 13.0, 17.0, 23.0, 37.0, 50.0]
    amplitudes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
    phases = [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi]
    
    for f in freqs:
        for A in amplitudes:
            for phi in phases:
                if time.time() > deadline:
                    return best_h
                h = _construct_cosine(f, A, phi)
                score = _compute_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Try some random structured sequences
    np.random.seed(42)
    for _ in range(500):
        if time.time() > deadline:
            break
        # Random frequency in [0, N/2]
        f = np.random.uniform(0, N/2)
        A = np.random.uniform(0, 0.5)
        phi = np.random.uniform(0, 2*np.pi)
        h = _construct_cosine(f, A, phi)
        score = _compute_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END

