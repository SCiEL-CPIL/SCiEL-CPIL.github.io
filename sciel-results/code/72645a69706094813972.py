"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Construct step heights using a spectral (sine-wave sum) representation.
    
    We build h as a sum of sine waves with random frequencies and phases,
    then normalize so that sum(h) = 1. The autoconvolution peak is bounded
    below by (sum h)^2 / N by Cauchy-Schwarz, giving value >= 2. This
    construction explores the space of oscillatory patterns to find cases
    where the autoconvolution maximum is close to this lower bound.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    best_value = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    # Try several random spectral constructions
    max_iters = 2000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random number of sine components
        n_components = rng.integers(1, 20)
        
        # Random frequencies (avoiding DC to keep oscillatory)
        freqs = rng.uniform(1.0, 50.0, size=n_components)
        phases = rng.uniform(0, 2 * np.pi, size=n_components)
        amplitudes = rng.uniform(0.5, 1.0, size=n_components)
        
        # Build the signal as sum of sine waves
        idx = np.arange(N)
        h = np.zeros(N)
        for f, p, a in zip(freqs, phases, amplitudes):
            h += a * np.sin(2 * np.pi * f * idx / N + p)
        
        # Add a small DC offset to ensure sum(h) != 0
        mean_h = np.mean(h)
        h += 0.1  # small positive DC
        
        # Normalize so sum(h) = 1
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        h = h / s
        
        # Compute the value
        autoconv = np.convolve(h, h, mode='full')
        max_ac = np.max(autoconv)
        value = abs(2 * N * max_ac / (np.sum(h) ** 2))
        
        if value < best_value:
            best_value = value
            best_h = h.copy()
        
        # Early stop if we hit the theoretical bound
        if best_value <= 2.0 + 1e-10:
            break
    
    # Fallback to seed if something went wrong
    if not np.isfinite(best_value) or best_value >= 2.0:
        # Try a few more with different strategies
        for _ in range(100):
            if time.time() > deadline:
                break
            # Random heights with nonzero mean
            h = rng.normal(0, 1, size=N) + 0.5
            s = np.sum(h)
            if abs(s) < 1e-10:
                continue
            h = h / s
            autoconv = np.convolve(h, h, mode='full')
            max_ac = np.max(autoconv)
            value = abs(2 * N * max_ac / (np.sum(h) ** 2))
            if value < best_value:
                best_value = value
                best_h = h.copy()
            if best_value <= 2.0 + 1e-10:
                break
    
    # Final safety: if still bad, return seed
    if not np.isfinite(best_value) or best_value >= 2.0 - 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
