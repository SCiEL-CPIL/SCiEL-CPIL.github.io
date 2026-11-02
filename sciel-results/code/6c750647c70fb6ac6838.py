"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconvolve_full(f):
    """Full autoconvolution of f via FFT (zero-padded)."""
    n = len(f)
    size = 2 * n - 1
    # Use next power of 2 for efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    ac = np.fft.irfft(F * np.conj(F), n=fft_size)
    return ac[:size]

def _score(f):
    """Compute max |autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconvolve_full(f)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s
    
    # Start with all-ones (theoretical optimum by Cauchy-Schwarz)
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    # Try sign patterns to see if we can do better (unlikely due to CS bound)
    # But maybe the task definition is different? Let's explore.
    # Random sign patterns
    rng = np.random.default_rng(42)
    
    for i in range(1000):
        if time.time() > deadline:
            break
        f = rng.choice([-1, 1], size=N).astype(float)
        sc = _score(f)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()
    
    # Try smooth step functions: sign changes at random positions
    for i in range(500):
        if time.time() > deadline:
            break
        n_steps = rng.integers(1, N)
        positions = np.sort(rng.choice(N, size=n_steps, replace=False))
        signs = rng.choice([-1, 1], size=n_steps)
        f = np.repeat(signs, np.diff(np.concatenate([[0], positions, [N]])))
        sc = _score(f)
        if sc < best_score:
            best_score = sc
            best_f = f.copy()
    
    return best_f
# EVOLVE-BLOCK-END

