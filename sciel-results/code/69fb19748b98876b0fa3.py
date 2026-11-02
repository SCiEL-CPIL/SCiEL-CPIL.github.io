"""Autoconvolution minimization via greedy coordinate descent with multiple restarts.

Structural change: replace simulated annealing with greedy single-bit flips.
Each flip is evaluated in O(N log N) via FFT, allowing thousands of flips within
the time budget. Multiple restarts from random initializations explore different
basins of attraction in the 2^400 sign space.

The metric is max_k |C(k)| where C(k) = sum_i x_i x_{i+k} is the linear
autoconvolution. Minimizing this directly targets the C3 upper bound.
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345


def _autoconv_max_abs(x):
    """Return max(|linear autoconvolution|) for a real array x of len N."""
    xf = np.zeros(L, dtype=np.float64)
    xf[:N] = x
    spec = np.fft.rfft(xf)
    conv = np.fft.irfft(spec * spec.conj(), n=L)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _greedy_coordinate_descent(x, t0, deadline):
    """Greedy single-bit flip coordinate descent. Returns (val, x)."""
    cur_val = _autoconv_max_abs(x)
    cur_x = x.copy()
    improved = True
    
    while improved and (time.time() - t0) < deadline:
        improved = False
        for i in range(N):
            if (time.time() - t0) > deadline:
                break
            # Try flipping position i
            cur_x[i] = -cur_x[i]
            new_val = _autoconv_max_abs(cur_x)
            if new_val < cur_val:
                cur_val = new_val
                improved = True
            else:
                cur_x[i] = -cur_x[i]  # revert
    
    return cur_val, cur_x


def construct():
    t0 = time.time()
    deadline = 110.0  # Safety margin before 150s kill

    rng = np.random.default_rng(_SEED)
    
    best_val = float('inf')
    best_x = None
    
    # Multiple restarts from random initializations
    n_restarts = 20
    for r in range(n_restarts):
        if (time.time() - t0) > deadline - 5.0:
            break
        
        # Random initialization
        x = rng.choice([-1.0, 1.0], size=N)
        
        # Run greedy coordinate descent
        val, xx = _greedy_coordinate_descent(x, t0, deadline - 5.0)
        
        if val < best_val:
            best_val = val
            best_x = xx
    
    return best_x
