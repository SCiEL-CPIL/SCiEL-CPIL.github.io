"""
Signed autocorrelation max-abs variant — sparse-poly + spectral-null design.

Strategy (structural change):
  * Represent the length-N=400 signal as a DC offset plus a small number of
    non-zero "spikes" (positions, magnitudes) AND/OR a few sinusoidal tones.
  * The metric M(x)=max_k|c_k|/(sum x)^2 is scale-invariant, so we normalize
    sum(x)=1 before scoring.
  * Use differential evolution over the low-dimensional parameter vector to
    create spectral nulls that suppress the off-diagonal autoconvolution.
  * Polish with L-BFGS-B on a smooth soft-max surrogate, then a few
    coordinate-descent / sign-flip local moves.
  * Explicit wall-clock deadline guard (return best before 140s).
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
DEADLINE = 138.0  # safety margin before 150s hard kill
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


def _norm(x):
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return x
    return x / s


def _sparse_to_signal(p, n_spikes):
    """
    p layout:
      p[0] = DC offset a0
      for each spike j in range(n_spikes):
        p[1+3*j]   = position (0..N-1)
        p[2+3*j]   = magnitude
    Signal is zero everywhere except at the spike positions (plus DC).
    """
    x = np.full(N, p[0])
    for j in range(n_spikes):
        pos = int(round(p[1 + 3 * j])) % N
        mag = p[2 + 3 * j]
        x[pos] += mag
    return x


def _sine_to_signal(p, K):
    """
    p layout:
      p[0] = a0 (DC)
      for each tone j in range(K):
        p[1+3*j]   = frequency bin (1..N-1)
        p[2+3*j]   = phase
        p[3+3*j]   = amplitude
    """
    i = np.arange(N)
    x = np.full(N, p[0])
    for j in range(K):
        base = 1 + 3 * j
        f = p[base]
        ph = p[base + 1]
        a = p[base + 2]
        x += a * np.cos(2.0 * np.pi * f * i / N + ph)
    return x


def _mixed_to_signal(p, n_spikes, K):
    """
    Combined: DC + n_spikes sparse + K sinusoidal tones.
    p layout:
      p[0] = a0
      next 3*n_spikes: spikes (pos, mag)  -- note: 2 params per spike, but we use 3 for uniformity (pos, mag, unused=0)
      Actually let's use 2 per spike: pos, mag. Then tones.
    """
    # Re-interpret: p[0]=a0, then for spikes: pos (0..N-1), mag
    # then for tones: freq, phase, amp
    x = np.full(N, p[0])
    idx = 1
    for j in range(n_spikes):
        pos = int(round(p[idx])) % N
        mag = p[idx + 1]
        x[pos] += mag
        idx += 2
    i = np.arange(N)
    for j in range(K):
        f = p[idx]
        ph = p[idx + 1]
        a = p[idx + 2]
        x += a * np.cos(2.0 * np.pi * f * i / N + ph)
        idx += 3
    return x


def _de_objective_sparse(p, n_spikes):
    x = _sparse_to_signal(p, n_spikes)
    x = _norm(x)
    return _score(x)


def _de_objective_sine(p, K):
    x = _sine_to_signal(p, K)
    x = _norm(x)
    return _score(x)


def _de_objective_mixed(p, n_spikes, K):
    x = _mixed_to_signal(p, n_spikes, K)
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


def _local_polish(x0, max_iter=300, tau=0.4):
    """L-BFGS-B on the 400-dim signal with a soft-max surrogate."""
    best_x = x0.copy()
    best_sc = _score(x0)

    def obj(x):
        if _elapsed() > DEADLINE:
            return best_sc
        return _softmax_abs_ac(x, tau=tau)

    try:
        res = minimize(obj, x0, method='L-BFGS-B', maxiter=max_iter,
                       options={'maxls': 30, 'ftol': 1e-12, 'gtol': 1e-9})
        if res.fun < best_sc:
            cand = res.x
            sc = _score(cand)
            if sc < best_sc:
                best_x = cand
                best_sc = sc
    except Exception:
        pass

    # Coordinate descent: try flipping signs and small perturbations
    if _elapsed() < DEADLINE - 10:
        x = best_x.copy()
        for it in range(200):
            if _elapsed() > DEADLINE - 10:
                break
            improved = False
            # Random subset of coordinates
            idx = np.random.choice(N, size=min(20, N), replace=False)
            for i in idx:
                # Try sign flip
                old = x[i]
                x[i] = -old
                sc = _score(x)
                if sc < best_sc:
                    best_sc = sc
                    best_x = x.copy()
                    improved = True
                else:
                    x[i] = old
                # Try small perturbations
                for delta in [0.01, -0.01, 0.05, -0.05]:
                    x[i] = old + delta
                    sc = _score(x)
                    if sc < best_sc:
                        best_sc = sc
                        best_x = x.copy()
                        improved = True
                        old = x[i]
                    else:
                        x[i] = old
            if not improved:
                break
        best_x = best_x

    return best_x


def construct():
    global _start
    _start = time.time()

    # Baseline: all-ones (score = 1.0)
    best_x = np.ones(N)
    best_score = _score(best_x)

    # --- Sparse spike optimization ---
    for n_spikes in [4, 6, 8, 10, 12]:
        if _elapsed() > DEADLINE - 30:
            break
        np.random.seed(2000 + n_spikes)
        bounds = [(-2.0, 2.0)]  # DC
        for j in range(n_spikes):
            bounds.append((0.0, float(N - 1)))  # position
            bounds.append((-3.0, 3.0))          # magnitude

        try:
            de_res = differential_evolution(
                _de_objective_sparse, bounds, args=(n_spikes,),
                maxiter=50, popsize=15, tol=1e-5,
                seed=2000 + n_spikes, init='sobol',
                polish=False, updating='deferred'
            )
            x_de = _norm(_sparse_to_signal(de_res.x, n_spikes))
            sc_de = _score(x_de)
            if sc_de < best_score:
                best_score = sc_de
                best_x = x_de

            if _elapsed() < DEADLINE - 20:
                x_ref = _local_polish(x_de, max_iter=200)
                sc_ref = _score(x_ref)
                if sc_ref < best_score:
                    best_score = sc_ref
                    best_x = x_ref
        except Exception:
            continue

    # --- Sine tone optimization ---
    for K in [3, 4, 5, 6]:
        if _elapsed() > DEADLINE - 25:
            break
        np.random.seed(3000 + K)
        bounds = [(-2.0, 2.0)]  # DC
        for j in range(K):
            bounds.append((1.0, float(N - 1)))   # frequency
            bounds.append((-np.pi, np.pi))       # phase
            bounds.append((-2.0, 2.0))           # amplitude

        try:
            de_res = differential_evolution(
                _de_objective_sine, bounds, args=(K,),
                maxiter=50, popsize=15, tol=1e-5,
                seed=3000 + K, init='sobol',
                polish=False, updating='deferred'
            )
            x_de = _norm(_sine_to_signal(de_res.x, K))
            sc_de = _score(x_de)
            if sc_de < best_score:
                best_score = sc_de
                best_x = x_de

            if _elapsed() < DEADLINE - 15:
                x_ref = _local_polish(x_de, max_iter=200)
                sc_ref = _score(x_ref)
                if sc_ref < best_score:
                    best_score = sc_ref
                    best_x = x_ref
        except Exception:
            continue

    # --- Mixed sparse + sine optimization ---
    for n_spikes, K in [(4, 2), (6, 3), (8, 2)]:
        if _elapsed() > DEADLINE - 20:
            break
        np.random.seed(4000 + n_spikes + K)
        bounds = [(-2.0, 2.0)]  # DC
        for j in range(n_spikes):
            bounds.append((0.0, float(N - 1)))  # position
            bounds.append((-3.0, 3.0))          # magnitude
        for j in range(K):
            bounds.append((1.0, float(N - 1)))   # frequency
            bounds.append((-np.pi, np.pi))       # phase
            bounds.append((-2.0, 2.0))           # amplitude

        try:
            de_res = differential_evolution(
                _de_objective_mixed, bounds, args=(n_spikes, K),
                maxiter=40, popsize=12, tol=1e-5,
                seed=4000 + n_spikes + K, init='sobol',
                polish=False, updating='deferred'
            )
            x_de = _norm(_mixed_to_signal(de_res.x, n_spikes, K))
            sc_de = _score(x_de)
            if sc_de < best_score:
                best_score = sc_de
                best_x = x_de

            if _elapsed() < DEADLINE - 12:
                x_ref = _local_polish(x_de, max_iter=200)
                sc_ref = _score(x_ref)
                if sc_ref < best_score:
                    best_score = sc_ref
                    best_x = x_ref
        except Exception:
            continue

    # --- Final polish on the best found ---
    if _elapsed() < DEADLINE - 8:
        x_final = _local_polish(best_x, max_iter=300, tau=0.3)
        sc_final = _score(x_final)
        if sc_final < best_score:
            best_x = x_final
            best_score = sc_final

    return best_x
