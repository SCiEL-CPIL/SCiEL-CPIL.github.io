"""
Signed autocorrelation max-abs variant — spectral (multi-sine) parameterization.

The signal x (length N=400) is written as
    x_i = a0 + sum_j a_j * cos(2*pi*f_j*i/N + ph_j)
so that its DFT energy is concentrated in a few bins. A flat-ish spectral
profile makes the linear autoconvolution's off-diagonal lags small by
construction (Wiener–Khinchin: autocorrelation is the IFFT of the power
spectrum). We minimize the metric
    M(x) = max_k |ac_k| / (sum x)^2
by:
  1. Differential evolution over the ~12–18 spectral parameters (global).
  2. L-BFGS-B on a smooth soft-max surrogate of the max-abs autocorrelation
     (local polish), with the DC offset fixed so the metric is scale-invariant
     (we always normalize sum(x)=1 before scoring).

Deadline guard: explicit wall-clock check, return best found before 140 s.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
DEADLINE = 140.0  # safety margin before 150 s hard kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _autoconv(x):
    """Linear (full) autoconvolution of x, lags -(N-1)..(N-1)."""
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    ac_full = np.fft.irfft(f * np.conj(f), n=2 * n)
    return np.concatenate([ac_full[n:], ac_full[:n]])


def _score(x):
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return 1e12
    ac = _autoconv(x)
    return float(np.max(np.abs(ac))) / (s * s)


def _params_to_signal(p, K):
    """p = [a0, f1, ph1, a1, f2, ph2, a2, ...] -> signal length N."""
    i = np.arange(N)
    x = np.full(N, p[0])
    for j in range(K):
        base = 1 + 3 * j
        f = p[base]
        ph = p[base + 1]
        a = p[base + 2]
        x += a * np.cos(2.0 * np.pi * f * i / N + ph)
    return x


def _norm(x):
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return x
    return x / s


def _de_objective(p, K):
    x = _params_to_signal(p, K)
    x = _norm(x)
    return _score(x)


def _softmax_abs_ac(x, tau):
    ac = _autoconv(x)
    a = np.abs(ac)
    amax = float(np.max(a))
    if amax < 1e-14:
        return 0.0
    sm = tau * (amax / tau + np.log(np.sum(np.exp((a - amax) / tau))))
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return 1e12
    return sm / (s * s)


def _local_polish(x0, max_iter=300):
    """L-BFGS-B on the 400-dim signal with a soft-max surrogate."""
    def obj(x):
        if _elapsed() > DEADLINE:
            return _score(x0)
        return _softmax_abs_ac(x, tau=0.4)
    try:
        res = minimize(obj, x0, method='L-BFGS-B', maxiter=max_iter,
                       options={'maxls': 30, 'ftol': 1e-12, 'gtol': 1e-9})
        if res.fun < _score(x0):
            return res.x
        return x0
    except Exception:
        return x0


def construct():
    global _start
    _start = time.time()

    # Baseline: all-ones (score = 1.0, since max|ac|=N, sum^2=N^2)
    best_x = np.ones(N)
    best_score = _score(best_x)

    # Try several spectral complexities K (number of sine components)
    for K in [3, 4, 5, 6, 8]:
        if _elapsed() > DEADLINE - 35:
            break

        np.random.seed(1234 + K)
        bounds = [(-1.0, 1.0)]  # a0 (DC)
        for j in range(K):
            bounds.append((1.0, N - 1.0))   # frequency (bin index, 1..N-1)
            bounds.append((-np.pi, np.pi))  # phase
            bounds.append((-2.0, 2.0))      # amplitude

        try:
            de_res = differential_evolution(
                _de_objective, bounds, args=(K,),
                maxiter=60, popsize=18, tol=1e-5,
                seed=1234 + K, init='sobol',
                polish=False, updating='deferred'
            )
        except Exception:
            continue

        x_de = _norm(_params_to_signal(de_res.x, K))
        sc_de = _score(x_de)
        if sc_de < best_score:
            best_score = sc_de
            best_x = x_de

        # Local polish on the 400-dim signal
        if _elapsed() > DEADLINE - 25:
            break
        x_ref = _local_polish(x_de, max_iter=250)
        sc_ref = _score(x_ref)
        if sc_ref < best_score:
            best_score = sc_ref
            best_x = x_ref

    # A few random spectral seeds + polish (cheap diversity)
    if _elapsed() < DEADLINE - 20:
        for trial in range(4):
            if _elapsed() > DEADLINE - 15:
                break
            np.random.seed(7000 + trial)
            K = 5
            p = [0.0]
            for j in range(K):
                p.append(np.random.uniform(2.0, N - 2.0))
                p.append(np.random.uniform(-np.pi, np.pi))
                p.append(np.random.uniform(-1.5, 1.5))
            x = _norm(_params_to_signal(p, K))
            x_ref = _local_polish(x, max_iter=150)
            sc = _score(x_ref)
            if sc < best_score:
                best_score = sc
                best_x = x_ref

    # Final high-precision polish on the current best
    if _elapsed() < DEADLINE - 8:
        x_final = _local_polish(best_x, max_iter=300)
        sc_final = _score(x_final)
        if sc_final < best_score:
            best_x = x_final
            best_score = sc_final

    return best_x
