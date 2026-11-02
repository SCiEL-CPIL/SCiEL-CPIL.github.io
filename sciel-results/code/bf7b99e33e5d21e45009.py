"""Search for a low-autocorrelation signed step function (400 equal intervals).

The C3 autoconvolution upper bound is governed by the peak aperiodic
autocorrelation sidelobe of the step heights (the main lobe is fixed for any
+-1 function, so the differentiating quantity is the sidelobe structure).
A signed (+-1) step function with low autocorrelation drives this well below
the all-ones baseline. We minimize it via simulated annealing (block-flip
moves) with greedy polishing and multiple restarts, under a hard deadline.
"""
import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-6
    start = time.time()
    while time.time() - start < time_budget:
        L = int(rng.integers(1, 12))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99995)
    return best_h, best


def _greedy_polish(h, time_budget, start):
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i < 0:
            break
        h[best_i] *= -1.0
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def construct():
    try:
        hard_deadline = 135.0
        start = time.time()
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        # Long initial anneal from a random +-1 start.
        budget = min(35.0, hard_deadline - (time.time() - start) - 5.0)
        if budget > 1.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Multiple shorter restarts for diversity.
        restart_id = 1
        while time.time() - start < hard_deadline - 5.0:
            sub = min(10.0, hard_deadline - (time.time() - start) - 5.0)
            if sub < 1.0:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1

        # Final greedy single-flip polish with whatever time remains.
        remaining = hard_deadline - (time.time() - start) - 1.0
        if remaining > 0.5:
            h, c = _greedy_polish(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
