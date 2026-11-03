"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _try_local_search(x, deadline, rng, max_rounds=50000):
    """Multi-flip local search with restarts."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Try flipping k random positions
        k = rng.integers(1, min(5, N//4) + 1)
        indices = rng.choice(N, size=k, replace=False)
        x_flipped = best_x.copy()
        x_flipped[indices] *= -1
        
        v = _metric(x_flipped)
        if v < best_val:
            best_val = v
            best_x = x_flipped
    
    return best_x, best_val


def _try_structured_patterns(deadline, rng):
    """Generate structured patterns and evaluate them."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Try periodic patterns
    for period in range(2, min(50, N//4 + 1)):
        if time.time() > deadline:
            break
        for phase in range(period):
            x = np.zeros(N)
            for i in range(N):
                if (i + phase) % period == 0:
                    x[i] = 1.0
                else:
                    x[i] = -1.0
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    
    # Try ramped patterns
    for step in [0.1, 0.2, 0.5, 1.0, 2.0]:
        if time.time() > deadline:
            break
        x = np.linspace(-1, 1, N) * step
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
    
    return best_x, best_val


def construct():
    # Deadline guard: return with a safety margin before the 150 s kill.
    deadline = time.time() + 130.0

    # Start from all-ones (Cauchy-Schwarz lower bound)
    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    # Phase 1: Try structured patterns
    if time.time() < deadline - 60:
        x_struct, val_struct = _try_structured_patterns(deadline - 60, rng)
        if val_struct < best_val:
            best_val = val_struct
            best_x = x_struct

    # Phase 2: Try random ±1 sequences
    if time.time() < deadline - 30:
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            cand = rng.choice([-1.0, 1.0], size=N)
            v = _metric(cand)
            if v < best_val:
                best_val = v
                best_x = cand.copy()

    # Phase 3: Local search from best candidate
    if time.time() < deadline - 10:
        best_x, best_val = _try_local_search(best_x, deadline - 10, rng, max_rounds=100000)

    return best_x
# EVOLVE-BLOCK-END
