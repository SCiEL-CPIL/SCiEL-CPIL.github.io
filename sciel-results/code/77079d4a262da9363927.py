"""Third autocorrelation inequality: flatten the autoconvolution spectrum.

Strategy (structural change vs. parent):
  * The objective 2*N*max(autoconv)/sum(h)^2 is scale-invariant, so we fix
    sum(h)=N and minimize max(autoconv).
  * autoconv(h) = IFFT(|FFT(h)|^2), so minimizing its max is equivalent to
    flattening |FFT(h)|^2.
  * We minimize a smooth max surrogate (logsumexp) with analytic gradients
    using L-BFGS-B.
  * Parameterization h = a*cos(phi) keeps the sum positive (avoids the
    zero-sum failure mode) while giving a rich, smooth search space.
  * Seeds: Zadoff-Chu (near-flat spectrum), chirp, and a constant.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 150.0
EPS = 1e-12
T = 0.03  # smoothness of logsumexp surrogate (in autoconv units, ~N scale)


def _value(h):
    s = np.sum(h)
    if abs(s) < EPS:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _smooth_obj_grad(x):
    """Smooth max(autoconv) surrogate + analytic gradient, sum fixed to N."""
    a = x[:N]
    phi = x[N:]
    h = a * np.cos(phi)
    h = h - np.mean(h) + 1.0  # project to sum=N (sum is invariant under this)
    H = np.fft.fft(h)
    p = np.abs(H) ** 2
    m = np.max(p)
    w = np.exp((p - m) / T)
    w = w / np.sum(w)
    val = m + T * np.log(np.sum(w))
    # dval/dp = w/T
    dval_dp = w / T
    # d|H|^2/dh = 2*Re(conj(H)*FFT( dh/dh_j )) = 2*conj(H)*conj(H) shifted
    # autoconv = IFFT(|H|^2); dval/dh = IFFT( dval_dp * |H|^2 )
    dval_dh = np.fft.ifft(dval_dp * p).real
    # chain rule through h = a*cos(phi)
    ga = dval_dh * np.cos(phi)
    gp = -dval_dh * a * np.sin(phi)
    return val, np.concatenate([ga, gp])


def _refine_direct(h, seconds):
    """Final refinement optimizing h directly (smooth surrogate, sum fixed)."""
    deadline = time.time() + seconds
    h0 = h - np.mean(h) + 1.0

    def f(x):
        hh = x - np.mean(x) + 1.0
        H = np.fft.fft(hh)
        p = np.abs(H) ** 2
        m = np.max(p)
        return m + T * np.log(np.sum(np.exp((p - m) / T)))

    def fg(x):
        hh = x - np.mean(x) + 1.0
        H = np.fft.fft(hh)
        p = np.abs(H) ** 2
        m = np.max(p)
        w = np.exp((p - m) / T)
        w = w / np.sum(w)
        val = m + T * np.log(np.sum(w))
        g = np.fft.ifft((w / T) * p).real
        return val, g

    try:
        res = minimize(fg, h0.copy(), method='L-BFGS-B', jac=True,
                       options={'maxiter': 6000, 'maxfun': 6000,
                                'ftol': 1e-15, 'gtol': 1e-9})
        if time.time() < deadline and res.x is not None:
            h0 = res.x
    except Exception:
        pass
    h0 = h0 - np.mean(h0) + 1.0
    return h0


def _zadoff_chu(a):
    h = np.exp(1j * (np.pi * a * np.arange(N) ** 2) / N).real
    return h


def _chirp():
    h = np.cos(2 * np.pi * (np.arange(N) ** 2) / (2 * N))
    return h


def construct():
    start = time.time()
    deadline = start + DEADLINE - 5.0  # safety margin

    def remaining():
        return deadline - time.time()

    # ---- Candidate starts (amplitude, phase) ----
    starts = []
    # Zadoff-Chu with several a
    for a in (2, 3, 4, 5, 6, 7, 8):
        h = _zadoff_chu(a)
        a_arr = np.abs(h) + 1e-3
        phi_arr = np.arctan2(0.0, np.ones_like(h))  # cos phase=0 -> h
        # better: represent h = a*cos(phi) with phi=0, a=h (h may be neg)
        a_arr = np.abs(h)
        phi_arr = np.where(h >= 0, 0.0, np.pi)
        starts.append((a_arr, phi_arr))
    # chirp
    h = _chirp()
    starts.append((np.abs(h), np.where(h >= 0, 0.0, np.pi)))
    # constant
    starts.append((np.ones(N), np.zeros(N)))

    best_h = np.ones(N)
    best_val = _value(best_h)

    # ---- L-BFGS-B on (a, phi) for each start, budget-split ----
    n_starts = len(starts)
    for idx, (a0, phi0) in enumerate(starts):
        if time.time() > deadline:
            break
        x0 = np.concatenate([a0, phi0])

        def fg(x):
            return _smooth_obj_grad(x)

        # per-start time budget (seconds), decreasing as we go
        per_start_budget = max(3.0, min(20.0, remaining() / max(1, n_starts - idx)))
        try:
            res = minimize(fg, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 20000, 'maxfun': 20000,
                                    'ftol': 1e-15, 'gtol': 1e-9})
        except Exception:
            res = None
        if res is not None and res.x is not None:
            h = res.x[:N] * np.cos(res.x[N:])
            h = h - np.mean(h) + 1.0
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # ---- Final direct refinement on the best h ----
    if time.time() < deadline - 1.0:
        best_h = _refine_direct(best_h, max(1.0, deadline - time.time() - 1.0))
        best_val = _value(best_h)

    # ---- Safety: ensure valid (finite, nonzero sum) ----
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    if abs(np.sum(best_h)) < EPS:
        best_h = best_h + 1.0
    return best_h
