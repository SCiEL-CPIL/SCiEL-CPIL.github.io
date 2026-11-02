"""Optimize a signed step function (400 intervals on [-1/4,1/4]) to minimize the
L-infinity norm of its autoconvolution (f * f).

METRIC CHOICE (explicit): The task notes that the paper and the published notebook
place the absolute value differently. We explicitly select the metric
    value(h) = max_t | (f * f)(t) |
i.e. the L-infinity norm of the *signed* autoconvolution (the absolute value is
applied to the autoconvolution result). This is the natural "upper bound on the
autoconvolution" and, unlike the autocorrelation (whose lag-0 is fixed at the
total area), its peak can be reduced by choosing the signs, which is why a
signed step function can improve the bound.

The continuous autoconvolution of the step function is, up to the constant interval
width Delta, the discrete linear autoconvolution c = h * h of the height vector h.
We therefore minimize max_k |c[k]| via simulated annealing with block-flip moves
(block flips of size 4-32 coordinate sign changes that reduce the autoconvolution
concentration more effectively than single-bit flips), followed by a greedy
single-bit polish. A wall-clock deadline guard returns the best h found.
"""
import time
import numpy as np

N = 400
FFT_N = 1024  # smallest power of two >= 2*N - 1 = 799 (linear autoconvolution length)


def _autoconv_peak(h):
    """Return max_k | (h * h)[k] | for the linear autoconvolution of h (length N)."""
    F = np.fft.rfft(h, n=FFT_N)
    c = np.fft.irfft(F * F, n=FFT_N)
    return float(np.max(np.abs(c[:2 * N - 1])))


def _greedy_polish(h, rng, time_limit=None, t0=None):
    """Greedy single-bit flips that reduce the autoconvolution peak."""
    cur = _autoconv_peak(h)
    improved = True
    sweeps = 0
    while improved:
        improved = False
        sweeps += 1
        for i in range(N):
            if time_limit is not None and (time.time() - t0) > time_limit:
                return h, cur
            h[i] *= -1.0
            v = _autoconv_peak(h)
            if v < cur:
                cur = v
                improved = True
            else:
                h[i] *= -1.0  # revert
    return h, cur


def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin: evaluator kills at 150 s; stop well before.
    POLISH_BUDGET = 8.0  # reserve this much wall-clock for the final greedy polish

    rng = np.random.default_rng(20240517)

    def rand_h():
        return rng.choice((-1.0, 1.0), size=N)

    # --- Initialization: best of several random starts ---
    best_h = None
    best_val = np.inf
    for _ in range(16):
        h = rand_h()
        v = _autoconv_peak(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    h = best_h.copy()
    cur_val = best_val

    # --- Simulated annealing with block-flip moves ---
    T_START = 40.0
    T_END = 0.02
    # Estimate total steps from the time budget (calibrated to ~30k steps/s).
    time_for_sa = max(1.0, DEADLINE - POLISH_BUDGET)
    est_steps = int(time_for_sa * 25000)  # conservative step-rate estimate
    cool = (T_END / T_START) ** (1.0 / max(1, est_steps))
    T = T_START
    REHEAT_T = 6.0
    REHEAT_AFTER = 4000  # steps without improvement before reheating

    block_sizes = (4, 8, 16, 32)
    no_improve = 0
    step = 0

    while (time.time() - t0) < (DEADLINE - POLISH_BUDGET):
        # choose a move
        r = rng.random()
        if r < 0.15:
            bs = int(rng.integers(2, 4))          # small block 2-3
        elif r < 0.85:
            bs = int(rng.choice(block_sizes))     # main block 4-32
        else:
            bs = int(rng.integers(33, 97))        # occasional large block 33-96
        if bs >= N:
            bs = N
        start = int(rng.integers(0, N - bs + 1))
        idx = np.arange(start, start + bs)

        h[idx] *= -1.0
        new_val = _autoconv_peak(h)
        delta = new_val - cur_val
        accept = (delta < 0.0) or (rng.random() < np.exp(-delta / max(T, 1e-12)))
        if accept:
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            h[idx] *= -1.0  # revert
            no_improve += 1

        step += 1
        # cool, with reheat on stagnation
        T *= cool
        if no_improve >= REHEAT_AFTER:
            T = max(T, REHEAT_T)
            no_improve = 0

    # --- Final greedy polish (on the best h found) ---
    polish_deadline = DEADLINE - 1.0  # leave a 1 s margin before the 120 s hard cap
    h = best_h.copy()
    h, v = _greedy_polish(h, rng, time_limit=polish_deadline, t0=t0)
    if v < best_val:
        best_val = v
        best_h = h.copy()

    # Safety: always return a valid signed step vector of length N.
    out = np.asarray(best_h, dtype=np.float64)
    if out.shape != (N,) or not np.all(np.isfinite(out)):
        out = rand_h()
    return out
