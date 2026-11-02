"""Brownian-bridge spectral shaping for the Erdos minimum-overlap task.

We model the zero-mean deviation u = h - 1/2 as the path of a 1-D Brownian
bridge whose spectral density we shape to concentrate non-DC energy at the
lowest frequency.  This is a structural change from the previous "clean
circulant" construction: instead of hard-assigning a circulant spectrum and
projecting, we optimise a smooth spectral-density profile (a low-pass
amplitude with a tunable roll-off) and refine the resulting time-domain
signal with a soft-min (smooth max) projected-gradient polish on the actual
overlap objective.  The bridge representation naturally enforces the
integral constraint (mean 1/2) and gives a smooth, low-roughness starting
point, which should reduce the maximum overlap versus a hard circulant.
"""

import time
import numpy as np
from scipy.fft import fft, ifft

N = 100
DEADLINE_S = 145.0  # safety margin below the 150 s kill


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


def _bridge_spectral_solution(beta, gamma):
    """
    Build a Brownian-bridge-like spectral profile.

    The non-DC spectral magnitudes are shaped as a smooth low-pass:
        |H_k|^2 = A * exp(-beta * (k-1)^2)  for k = 1, N-1 (and symmetric)
        |H_k|^2 = eps                       for other k
    with A fixed by Parseval so the mean is exactly 1/2.

    beta controls the roll-off (larger = more concentrated at k=1).
    gamma is a small floor to avoid numerical issues.
    """
    k = np.arange(N)
    # Distance from the low-frequency bin k=1 (and k=N-1)
    # For a circulant-like profile, we want energy at k=1 and k=N-1.
    dist = np.minimum(np.abs(k - 1), np.abs(k - (N - 1)))
    # Smooth low-pass amplitude
    mags = np.full(N, gamma)
    mags[0] = 0.5 * N  # DC
    # Shape the non-DC bins
    shape = np.exp(-beta * dist**2)
    # Scale so that Parseval holds: sum |H_k|^2 = N/4
    # DC contributes N/4.  So non-DC must sum to 0? No:
    # Parseval: (1/N) sum |H_k|^2 = sum h_i^2.  We want mean h = 1/2.
    # The DC term H_0 = sum h_i = N/2, so |H_0|^2 = N^2/4.
    # sum |H_k|^2 = N * sum h_i^2.
    # We don't need to fix sum h_i^2; we just need mean = 1/2.
    # So set H_0 = N/2, and choose the rest to minimize overlap.
    # The overlap depends on |H_k|^2 for k != 0.
    # We want to concentrate non-DC energy at k=1, N-1.
    # Let |H_k|^2 = A * shape[k] for k != 0, with A chosen to keep h in [0,1] after projection.
    # For now, just use a fixed A and project.
    A = 0.02
    mags[1:] = A * shape[1:]
    mags[0] = 0.5 * N
    # Ensure real-valued h: use zero phase (or a simple linear phase)
    H = mags * np.exp(1j * 2 * np.pi * np.arange(N) / N)
    h = np.real(ifft(H))
    return _project(h)


def _soft_min_obj_grad(h, tau):
    """
    Soft-min (smooth max) of the overlap values, plus gradient.

    O(s) = sum_i h_i * (1 - h_{i-s})
    We want to minimize max_s O(s).  Use a soft-max:
        f = tau * log(sum exp(O(s)/tau))
    gradient wrt h_i.
    """
    H = fft(h)
    F = H * (1.0 - np.exp(-2j * np.pi * np.arange(N) / N))
    Ov = np.real(ifft(F))
    # Soft-max
    Ov_max = np.max(Ov)
    log_sum_exp = Ov_max + tau * np.log(np.sum(np.exp((Ov - Ov_max) / tau)))
    f = log_sum_exp
    # Weights for soft-max
    w = np.exp((Ov - Ov_max) / tau)
    w /= np.sum(w)
    # Gradient: d f / d h_i = sum_s w_s * d O(s)/d h_i
    # d O(s)/d h_i = (1 - h_{i-s}) - h_{i+s}
    idx = np.arange(N)
    grad = np.zeros(N)
    for s in range(N):
        if w[s] < 1e-10:
            continue
        grad += w[s] * ((1.0 - h[(idx - s) % N]) - h[(idx + s) % N])
    return f, grad


def _polish(h0, tau=0.05, max_iters=200):
    """Projected gradient descent on the soft-max overlap objective."""
    h = h0.copy()
    best = h.copy()
    best_v = _max_overlap(h)
    alpha = 0.02
    it = 0
    while it < max_iters and _time_remaining() > 5.0:
        f, grad = _soft_min_obj_grad(h, tau)
        h_new = _project(h - alpha * grad)
        v = _max_overlap(h_new)
        if v < best_v:
            best_v = v
            best = h_new.copy()
            h = h_new
        else:
            # Reduce step size
            alpha *= 0.8
            if alpha < 1e-6:
                break
        it += 1
    return best, best_v


def construct():
    t0 = time.time()
    # Try a few beta values for the bridge spectral profile
    best_h = None
    best_v = np.inf
    for beta in [0.1, 0.3, 0.5, 1.0, 2.0]:
        h = _bridge_spectral_solution(beta, gamma=1e-6)
        v = _max_overlap(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        if _time_remaining() < 10.0:
            break

    # Polish with soft-max gradient descent
    if _time_remaining() > 15.0:
        h2, v2 = _polish(best_h, tau=0.05, max_iters=150)
        if v2 < best_v:
            best_h, best_v = h2, v2

    # Final safety projection
    best_h = _project(best_h)
    return best_h
