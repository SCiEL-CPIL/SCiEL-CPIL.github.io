"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_l2(h):
    """Compute L2 norm of the autoconvolution of h using FFT."""
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)[:2*n-1]
    return float(np.sum(ac ** 2))

def _try_local_search(h, budget):
    """Local search: flip signs of individual elements to reduce L2."""
    best = h.copy()
    best_val = _autoconv_l2(best)
    n = len(best)
    while time.time() < budget:
        improved = False
        for i in range(n):
            if time.time() > budget:
                break
            old = best[i]
            best[i] = -best[i]
            val = _autoconv_l2(best)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best[i] = old
        if not improved:
            break
    return best, best_val

def _random_signed(n, rng):
    """Generate random signed heights in {-1, 1}."""
    return rng.choice([-1.0, 1.0], size=n)

def construct():
    deadline = time.time() + 110
    best_h = None
    best_val = float('inf')
    rng = np.random.default_rng(42)
    
    n_restarts = 100
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        h = _random_signed(N, rng)
        h, val = _try_local_search(h, deadline)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        return np.ones(N)
    return best_h
# EVOLVE-BLOCK-END
