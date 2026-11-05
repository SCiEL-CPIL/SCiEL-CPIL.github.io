"""Two-level + smooth-max surrogate: differential evolution over low-dim params."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import softmax

N = 400
DEADLINE = 148.0


def _autocorr_fft(h):
    n = len(h)
    nfft = 2 ** (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    ac = np.fft.irfft(H * np.conj(H), n=nfft)[:2 * n - 1]
    return ac


def _value(h):
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf
    c = _autocorr_fft(h)
    return abs(2.0 * N * float(np.max(c)) / (S * S))


def _smooth_max_value(h, tau=50.0):
    """Smooth surrogate: replace max with softmax-weighted peak."""
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf
    c = _autocorr_fft(h)
    # Use a temperature-based smooth max
    c_scaled = c / tau
    # Clip for numerical stability
    c_scaled = np.clip(c_scaled, -500, 500)
    w = np.exp(c_scaled - np.max(c_scaled))
    w /= np.sum(w)
    smooth_max = np.sum(w * c)
    return abs(2.0 * N * smooth_max / (S * S))


def _h_from_params(params):
    """
    Parameterize h as: h[i] = a + b*cos(2*pi*f1*i/N + p1) + c*cos(2*pi*f2*i/N + p2)
    + d*cos(2*pi*f3*i/N + p3)
    7 parameters: a, b, c, d, f1, p1, f2, p2, f3, p3 -> 10 params
    """
    a, b, c, d, f1, p1, f2, p2, f3, p3 = params
    i = np.arange(N, dtype=float)
    h = (a
         + b * np.cos(2.0 * np.pi * f1 * i / N + p1)
         + c * np.cos(2.0 * np.pi * f2 * i / N + p2)
         + d * np.cos(2.0 * np.pi * f3 * i / N + p3))
    return h


def _obj(params, tau=50.0):
    h = _h_from_params(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    return _smooth_max_value(h, tau=tau)


def _obj_exact(params):
    h = _h_from_params(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    return _value(h)


def construct():
    t0 = time.time()

    # Bounds for the 10 parameters
    # a: mean level, keep positive sum -> a in [0.5, 3.0]
    # b,c,d: amplitudes in [-2, 2]
    # f1,f2,f3: frequencies in [0.1, 5.0] (cycles over the sequence)
    # p1,p2,p3: phases in [0, 2*pi]
    bounds = [
        (0.3, 3.0),   # a
        (-2.0, 2.0),  # b
        (-2.0, 2.0),  # c
        (-2.0, 2.0),  # d
        (0.1, 5.0),   # f1
        (0.0, 2 * np.pi),  # p1
        (0.1, 5.0),   # f2
        (0.0, 2 * np.pi),  # p2
        (0.1, 5.0),   # f3
        (0.0, 2 * np.pi),  # p3
    ]

    # Differential evolution for global search
    np.random.seed(123)
    best_params = None
    best_v = np.inf

    # Phase 1: Differential evolution with smooth max
    if time.time() - t0 < DEADLINE - 30:
        try:
            de_res = differential_evolution(
                _obj,
                bounds,
                args=(30.0,),
                seed=123,
                maxiter=80,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol',
            )
            if de_res.x is not None:
                v = _obj_exact(de_res.x)
                if v < best_v:
                    best_v = v
                    best_params = de_res.x.copy()
        except Exception:
            pass

    # Phase 2: Local refinement with Nelder-Mead using exact value
    if best_params is not None and time.time() - t0 < DEADLINE - 5:
        try:
            nm_res = minimize(
                _obj_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            if nm_res.x is not None:
                v = _obj_exact(nm_res.x)
                if v < best_v:
                    best_v = v
                    best_params = nm_res.x.copy()
        except Exception:
            pass

    # Phase 3: Additional DE restarts if time permits
    if time.time() - t0 < DEADLINE - 15:
        for seed_val in [456, 789]:
            if time.time() - t0 > DEADLINE - 10:
                break
            try:
                de_res2 = differential_evolution(
                    _obj,
                    bounds,
                    args=(20.0,),
                    seed=seed_val,
                    maxiter=40,
                    popsize=12,
                    tol=1e-8,
                    polish=False,
                    init='latinhypercube',
                )
                if de_res2.x is not None:
                    v = _obj_exact(de_res2.x)
                    if v < best_v:
                        best_v = v
                        best_params = de_res2.x.copy()
            except Exception:
                pass

    # Safety: if we didn't find anything good, try a simple heuristic
    if best_params is None or best_v >= 2.0:
        # Try a simple alternating pattern with slight asymmetry
        h_alt = np.ones(N)
        h_alt[::2] = -0.5  # every other is -0.5
        v_alt = _value(h_alt)
        if v_alt < best_v:
            best_v = v_alt
            # Just return this directly
            return h_alt

    if best_params is None:
        return np.ones(N)

    h = _h_from_params(best_params)

    # Final safety check
    if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-10:
        return np.ones(N)

    # Verify with exact value
    v_final = _value(h)
    if v_final >= 2.0:
        return np.ones(N)

    return h
