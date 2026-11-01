"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_sidelobe(x):
    """Compute max |R(k)| for k=1..N-1 where R(k) is autocorrelation."""
    N = len(x)
    # R(k) = sum_{i=0}^{N-k-1} x[i] * x[i+k]
    max_val = 0
    for k in range(1, N):
        s = np.dot(x[:N-k], x[k:])
        if abs(s) > max_val:
            max_val = abs(s)
    return max_val


def _try_flip(x, i):
    """Return new x with bit i flipped, and the new max sidelobe."""
    x_new = x.copy()
    x_new[i] = -x_new[i]
    return x_new, _max_sidelobe(x_new)


def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Start with random ±1 sequence
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N)
    best_x = x.copy()
    best_val = _max_sidelobe(x)
    
    # Local search: flip individual bits
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            x_new, val_new = _try_flip(x, i)
            if val_new < best_val:
                x = x_new
                best_x = x_new.copy()
                best_val = val_new
                improved = True
    
    return best_x
# EVOLVE-BLOCK-END
