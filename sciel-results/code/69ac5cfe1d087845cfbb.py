"""Chirp-based design: expanded family with linear + quadratic phase, plus direct h-polish."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
DEADLINE = 145.0  # safety margin inside the 150 s kill

_i = np.arange(N, dtype=float)
_i_centered = _i - N / 2.0


def _value(h):
    S = float(np.sum(h))
    if S == 0.0:
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(c)) / (S * S))


def _h_from_params3(p):
    """h[i] = cos(phi + a*i + k*i^2)"""
    phi, a, k = p
    return np.cos(phi + a * _i + k * _i * _i)


def _h_from_params_centered(p):
    """h[i] = cos(phi + k*(i-N/2)^2)"""
    phi, k = p
    return np.cos(phi + k * _i_centered * _i_centered)


def _obj3(p):
    return _value(_h_from_params3(p))


def _obj_centered(p):
    return _value(_h_from_params_centered(p))


def _direct_obj(h):
    """Objective for direct h optimization; penalize if sum is near zero."""
    S = float(np.sum(h))
    if abs(S) < 1e-6:
        return 1e10
    c = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(c)) / (S * S))


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_v = _value(best_h)

    # Phase 1: DE over 3-param chirp family (phi, a, k)
    bounds3 = [(-np.pi, np.pi), (-0.1, 0.1), (-0.05, 0.05)]
    try:
        res = differential_evolution(
            _obj3,
            bounds3,
            seed=0,
            maxiter=30,
            tol=1e-8,
            popsize=10,
            init='sobol',
            polish=False,
            update_interval=1,
        )
        if res.x is not None:
            h = _h_from_params3(res.x)
            v = _value(h)
            if v < best_v:
                best_v = v
                best_h = h
    except Exception:
        pass

    # Phase 2: DE over centered chirp family (phi, k)
    if time.time() - t0 < DEADLINE - 40:
        bounds_c = [(-np.pi, np.pi), (-0.05, 0.05)]
        try:
            res_c = differential_evolution(
                _obj_centered,
                bounds_c,
                seed=1,
                maxiter=25,
                tol=1e-8,
                popsize=10,
                init='sobol',
                polish=False,
            )
            if res_c.x is not None:
                h = _h_from_params_centered(res_c.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h
        except Exception:
            pass

    # Phase 3: Local Nelder-Mead refinement on 3-param chirp from best found
    if time.time() - t0 < DEADLINE - 20:
        # Find best 3-param point by re-running a short DE with different seed
        try:
            res2 = differential_evolution(
                _obj3,
                bounds3,
                seed=42,
                maxiter=20,
                tol=1e-8,
                popsize=8,
                init='sobol',
                polish=False,
            )
            if res2.x is not None:
                p_init = res2.x
            else:
                p_init = (0.0, 0.0, 0.0)
            
            res_nm = minimize(
                _obj3,
                p_init,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 800},
            )
            if res_nm.x is not None:
                h = _h_from_params3(res_nm.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h
        except Exception:
            pass

    # Phase 4: Direct Nelder-Mead polish on h itself
    if time.time() - t0 < DEADLINE - 10:
        try:
            h_init = best_h.copy()
            # Normalize to have sum ~ N (keeps scale reasonable)
            S = float(np.sum(h_init))
            if abs(S) > 1e-6:
                h_init = h_init * (N / S)
            
            res_h = minimize(
                _direct_obj,
                h_init,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 300, 'maxfev': 500},
            )
            if res_h.x is not None:
                h = res_h.x
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h
        except Exception:
            pass

    # Safety: ensure we return before the deadline
    if time.time() - t0 > DEADLINE - 2:
        return np.ones(N)

    return best_h
