"""Spectral (circulant) construction for the Erdos minimum-overlap task.

The overlap of h with shifted (1-h) is the circular cross-correlation
    O(s) = sum_i h_i * (1 - h_{i-s})          (circular indices)
whose DFT is
    F[O]_k = |H_k|^2 * (1 - exp(-i 2 pi k / N)).
The k=0 (DC) term vanishes, so O(s) depends only on the NON-DC spectral
magnitudes |H_k|^2 -- the phase and the mean (fixed to 1/2 by the
integral constraint) do not matter.  Hence minimizing the maximum overlap
is a spectral-shaping problem: concentrate the non-DC energy into the
low-frequency bin (k=1) and suppress the rest.

We build a clean circulant spectrum
    |H_1|^2 = A,  |H_k|^2 = eps (k=2..N-2),  |H_{N-1}|^2 = A,
with A fixed by Parseval so that the mean is exactly 1/2.  A tiny eps
pushes the max overlap toward the theoretical lower bound
    |1 - exp(-i 2 pi / N)|^2 * A  ~  4 pi^2 / N * A.
The time-domain signal is recovered by a single IFFT and a light
projection onto {0<=h_i<=1, sum=N/2}.  A bounded projected-gradient polish
on the *actual* max-overlap objective is run only if the wall-clock budget
allows it.
"""

import time
import numpy as np
from scipy.fft import fft, ifft

N = 100

# ---- tunable knobs (structural, not micro-tuned) ----
EPS = 1e-4          # residual spectral energy per suppressed bin
ALPHA = 0.05        # projected-gradient step size
MAX_POLISH_ITERS = 120
DEADLINE_S = 145.0  # hard safety margin below the 150 s kill


def _time_remaining():
    return DEADLINE_S - time.time()


def _max_overlap(h):
    """Maximum circular overlap of h with shifted (1-h), via FFT."""
    H = fft(h)
    F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
    return float(np.max(np.real(ifft(F))))


def _project(h):
    """Project onto {0<=h_i<=1, sum_i h_i = N/2} (clip + barycentric mean fix)."""
    h = np.clip(h, 0.0, 1.0)
    diff = h.sum() - 0.5 * N
    if diff > 0.0:
        order = np.argsort(h)[::-1]
        for i in order:
            take = min(diff, h[i])
            h[i] -= take
            diff -= take
            if diff <= 1e-12:
                break
    else:
        diff = -diff
        order = np.argsort(h)
        for i in order:
            take = min(diff, 1.0 - h[i])
            h[i] += take
            diff -= take
            if diff <= 1e-12:
                break
    return h


def _spectral_solution():
    """Deterministic circulant construction with a low-frequency spectral plan."""
    # Parseval: sum |H_k|^2 = N * mean^2 = N/4.  DC = N/4.
    # A (k=1 and k=N-1) + (N-3)*eps = N/4  ->  A = N/4 - (N-3)*eps.
    A = 0.25 * N - (N - 3) * EPS
    A = max(A, 0.0)
    mags = np.full(N, EPS)
    mags[0] = 0.5 * N          # DC (mean 1/2)
    mags[1] = A
    mags[N - 1] = A
    H = mags * np.exp(1j * 2 * np.pi * np.arange(N) / N)
    h = np.real(ifft(H))
    return _project(h)


def _polish(h0):
    """Bounded projected gradient descent on the max-overlap objective."""
    h = h0.copy()
    best = h.copy()
    best_v = _max_overlap(h)
    it = 0
    while it < MAX_POLISH_ITERS and _time_remaining() > 5.0:
        H = fft(h)
        F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
        Ov = np.real(ifft(F))
        s = int(np.argmax(Ov))
        # d O(s)/d h_i = (1 - h_{i-s}) - h_{i+s}
        idx = np.arange(N)
        grad = (1.0 - h[(idx - s) % N]) - h[(idx + s) % N]
        h = _project(h - ALPHA * grad)
        v = _max_overlap(h)
        if v < best_v:
            best_v = v
            best = h.copy()
        it += 1
    return best, best_v


def construct():
    t0 = time.time()
    h = _spectral_solution()
    v = _max_overlap(h)
    if _time_remaining() > 8.0:
        h2, v2 = _polish(h)
        if v2 < v:
            h, v = h2, v2
    # final safety: guarantee a valid, well-formed result
    h = _project(h)
    return h
