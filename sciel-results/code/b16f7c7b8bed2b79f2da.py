"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _value(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / s**2)


def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Try spectral construction: sum of sinusoids with zero mean
    # to reduce autoconvolution peak relative to sum^2
    rng = np.random.default_rng(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate random frequencies and amplitudes
        n_freqs = rng.integers(3, 15)
        freqs = rng.uniform(1, N/2, size=n_freqs)
        amps = rng.uniform(-1, 1, size=n_freqs)
        phases = rng.uniform(0, 2*np.pi, size=n_freqs)
        
        h = np.zeros(N)
        for f, a, p in zip(freqs, amps, phases):
            h += a * np.sin(2*np.pi*f*np.arange(N)/N + p)
        
        # Enforce zero-sum by subtracting mean
        h -= np.mean(h)
        
        # Scale to have reasonable magnitude
        if np.sum(h) == 0:
            continue
        h = h / np.sum(h) * 1.0  # normalize sum to 1
        
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Local optimization: try small perturbations around best
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        h = best_h.copy()
        # Random perturbation
        idx = rng.integers(0, N, size=5)
        h[idx] += rng.normal(0, 0.1, size=5)
        # Re-normalize sum
        s = np.sum(h)
        if s == 0:
            continue
        h = h / s
        
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
