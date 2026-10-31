"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3(x):
    """Compute C3 bound: integral of |autocorrelation|^3 dt"""
    autocorr = np.correlate(x, x, mode='full')
    dt = 1.0/N
    return np.sum(np.abs(autocorr)**3) * dt


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initialize with all +1
    x = np.ones(N, dtype=np.float64)
    best_c3 = _compute_c3(x)
    
    # Try some simple patterns
    patterns = [
        np.ones(N),
        -np.ones(N),
        np.tile([1, -1], N//2),
        np.tile([1, 1, -1, -1], N//4),
        np.tile([1, -1, -1, 1], N//4),
    ]
    
    for p in patterns:
        if len(p) == N:
            c3 = _compute_c3(p)
            if c3 < best_c3:
                best_c3 = c3
                x = p.copy()
    
    # Local search: flip individual intervals
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            c3 = _compute_c3(x)
            if c3 < best_c3 - 1e-15:
                best_c3 = c3
                improved = True
            else:
                x[i] = -x[i]  # Revert
    
    return x
# EVOLVE-BLOCK-END
