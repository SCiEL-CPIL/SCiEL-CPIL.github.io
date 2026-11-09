"""Rewritten program: local search for signed step heights minimizing
max absolute off-zero autoconvolution divided by squared integral."""

import time
import numpy as np

N = 400
DEADLINE = 110.0  # seconds wall-clock safety margin before 120s kill


def _autocorr_full(f):
    """Full (linear) autocorrelation via FFT. Returns array of length 2N-1
    centered such that index N-1 is lag 0."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, size)[:2 * n - 1]
    # reorder to lags -(n-1)..(n-1)
    ac = np.concatenate([ac[n - 1:], ac[:n - 1]])
    return ac


def _objective(f):
    """max absolute off-zero autoconvolution divided by (sum f)^2.
    If sum is ~0, return a large penalty."""
    s = f.sum()
    s2 = s * s
    if abs(s2) < 1e-9:
        return float('inf')
    ac = _autocorr_full(f)
    # ac index n-1 is lag 0; exclude it
    n = len(f)
    off = np.concatenate([ac[:n - 1], ac[n:]])
    return float(np.max(np.abs(off))) / s2


def construct():
    start = time.time()
    f = np.ones(N, dtype=np.float64)
    best = f.copy()
    best_obj = _objective(best)

    # Precompute FFT size
    n = N
    size = 1
    while size < 2 * n - 1:
        size <<= 1

    def ac_offmax(f):
        F = np.fft.rfft(f, size)
        power = F * np.conj(F)
        ac = np.fft.irfft(power, size)[:2 * n - 1]
        off = np.concatenate([ac[:n - 1], ac[n:]])
        return np.max(np.abs(off))

    current = f.copy()
    cur_offmax = ac_offmax(current)
    cur_sum = current.sum()

    # Greedy single-flip local search
    improved = True
    while improved:
        if time.time() - start > DEADLINE:
            break
        improved = False
        # Try flipping each position; pick the best improving flip
        best_gain = 0.0
        best_idx = -1
        for i in range(n):
            if time.time() - start > DEADLINE:
                break
            new_sum = cur_sum - 2.0 * current[i]
            if abs(new_sum) < 1e-9:
                continue
            # apply flip
            old = current[i]
            current[i] = -old
            offmax = ac_offmax(current)
            current[i] = old
            obj = offmax / (new_sum * new_sum)
            gain = best_obj - obj
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx >= 0:
            old = current[best_idx]
            current[best_idx] = -old
            cur_sum -= 2.0 * old
            cur_offmax = ac_offmax(current)
            new_obj = cur_offmax / (cur_sum * cur_sum)
            best_obj = new_obj
            best = current.copy()
            improved = True

    if time.time() - start > DEADLINE:
        return best

    # Second pass: random restarts / multi-start if time remains
    rng = np.random.default_rng(0)
    for _ in range(200):
        if time.time() - start > DEADLINE:
            break
        cand = rng.choice([-1.0, 1.0], size=n)
        s = cand.sum()
        if abs(s) < 1e-9:
            continue
        obj = ac_offmax(cand) / (s * s)
        if obj < best_obj:
            best_obj = obj
            best = cand.copy()
            # local refine
            cur = cand.copy()
            cur_sum = s
            for _pass in range(3):
                if time.time() - start > DEADLINE:
                    break
                prog = False
                for i in range(n):
                    if time.time() - start > DEADLINE:
                        break
                    ns = cur_sum - 2.0 * cur[i]
                    if abs(ns) < 1e-9:
                        continue
                    old = cur[i]
                    cur[i] = -old
                    om = ac_offmax(cur)
                    cur[i] = old
                    o = om / (ns * ns)
                    if o < best_obj - 1e-12:
                        cur[i] = -old
                        cur_sum = ns
                        best_obj = o
                        best = cur.copy()
                        prog = True
                if not prog:
                    break

    return best
