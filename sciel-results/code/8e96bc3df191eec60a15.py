"""Rewritten program: binary sign-sequence search + continuous polish for C3'."""
import numpy as np
from numpy.fft import fft, ifft
import time

N = 400

# Precompute FFT size (power of 2 >= 2N-1) for fast autoconvolution
_SIZE = 1
while _SIZE < 2 * N - 1:
    _SIZE *= 2


def _autoconv(x):
    """Linear autoconvolution via FFT (returns full length 2N-1)."""
    F = fft(x, _SIZE)
    A = ifft(F * np.conj(F)).real
    return A[:2 * N - 1]


def _maxabs_autoconv(x):
    """max |autoconv(x)|."""
    F = fft(x, _SIZE)
    A = ifft(F * np.conj(F)).real[:_SIZE]
    return np.max(np.abs(A))


def _score(x):
    """max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return _maxabs_autoconv(x) / (s * s)


def _score_binary(s):
    """Score for a +/-1 sign sequence; integral is constant (N^2)."""
    return _maxabs_autoconv(s) / (N * N)


# Basis autoconvolution vectors: B[i] = autoconv(e_i), shape (2N-1, N)
_BASIS = None


def _basis():
    global _BASIS
    if _BASIS is None:
        _BASIS = np.zeros((2 * N - 1, N))
        for i in range(N):
            e = np.zeros(N)
            e[i] = 1.0
            _BASIS[:, i] = _autoconv(e)
    return _BASIS


def _greedy_signs(s, budget):
    """Greedy coordinate-flip descent on a +/-1 sign sequence."""
    cur = s.copy()
    cur_sc = _score_binary(cur)
    B = _basis()
    end = time.time() + budget
    while time.time() < end:
        # autoconv of cur
        A0 = _autoconv(cur)
        # new autoconv if flipping coordinate i: A0 - 2*cur[i]*B[:,i]
        # cost_i = max_k |A0[k] - 2*cur[i]*B[k,i]|
        M = A0[:, None] - 2.0 * (cur[None, :] * B)
        costs = np.max(np.abs(M), axis=0)
        best_i = int(np.argmin(costs))
        if costs[best_i] + 1e-12 < cur_sc * N * N:
            cur[best_i] = -cur[best_i]
            cur_sc = costs[best_i] / (N * N)
        else:
            # local min; try a small batch of simultaneous flips
            k = np.random.default_rng().integers(2, 8)
            cand = cur.copy()
            flip = np.random.default_rng().choice(N, size=k, replace=False)
            cand[flip] = -cand[flip]
            cs = _score_binary(cand)
            if cs < cur_sc - 1e-12:
                cur = cand
                cur_sc = cs
            else:
                break
    return cur, cur_sc


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s
    rng = np.random.default_rng(7)

    # ---------- Phase 1: binary sign-sequence search ----------
    best = np.ones(N)
    best_sc = _score(best)

    for trial in range(6):
        if time.time() > deadline - 40.0:
            break
        s = rng.choice([-1.0, 1.0], size=N)
        s2, _ = _greedy_signs(s, 25.0)
        sc_full = _score(s2)
        if sc_full < best_sc:
            best_sc = sc_full
            best = s2.copy()

    sc_ones = _score(np.ones(N))
    if sc_ones < best_sc:
        best_sc = sc_ones
        best = np.ones(N)

    # ---------- Phase 2: continuous-height polish ----------
    x = best.astype(float).copy()
    sc = _score(x)
    end_polish = deadline - 5.0
    step = 0.05
    while time.time() < end_polish and step > 1e-4:
        improved = False
        for _ in range(400):
            if time.time() > end_polish:
                break
            n_p = rng.integers(1, 6)
            idx = rng.choice(N, size=n_p, replace=False)
            x_new = x.copy()
            x_new[idx] += rng.normal(0, step, n_p)
            sc_new = _score(x_new)
            if sc_new < sc - 1e-10:
                x = x_new
                sc = sc_new
                improved = True
        if not improved:
            step *= 0.5

    if sc < best_sc:
        best_sc = sc
        best = x.copy()

    return best
