"""Uncertainty inequality: tune Hermite-Gaussian to minimize A(g)*A(F[g]).

Fourier invariance makes A(F[g]) = A(g), so we minimize A(g)^2.
g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) e^{-x^2/2}, a12 fixed by g(0)=0.
A(g) is a smooth function of (a0,a4,a8); we do a high-precision global search.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import quad, simpson
from scipy.optimize import minimize, basinhopping

_T0 = time.time()
_DEADLINE = 200.0  # safety margin below the 210 s kill


def _hermite(n):
    """Probabilists' Hermite polynomial H_n(x) (monic, H_0=1,H_1=x,H_2=x^2-1)."""
    p = np.array([1.0])
    for k in range(1, n + 1):
        p = k * np.poly1d([1.0, 0.0]) * p + (k - 1) * np.poly1d(p) * np.poly1d([1.0, 0.0]) * 0 + (k - 1) * np.poly1d(p)
        # correct recurrence: H_k = x*H_{k-1} - (k-1)*H_{k-2}
    # recompute cleanly
    H = [np.poly1d([1.0]), np.poly1d([1.0, 0.0])]
    for k in range(2, n + 1):
        H.append(np.poly1d([1.0, 0.0]) * H[k - 1] - (k - 1) * H[k - 2])
    return H[n]


_H0 = _hermite(0)
_H4 = _hermite(4)
_H8 = _hermite(8)
_H12 = _hermite(12)

# g(0) = a0 + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0  =>  a12 = -(a0 + a4*H4(0) + a8*H8(0))/H12(0)
_H4_0 = float(_H4(0.0))
_H8_0 = float(_H8(0.0))
_H12_0 = float(_H12(0.0))


def _coeffs(a0, a4, a8):
    a12 = -(a0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    return a0, a4, a8, a12


def _poly(a0, a4, a8, a12):
    return a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12


def _density(a0, a4, a8, x):
    a12 = -(a0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    p = (a0 * _H0(x) + a4 * _H4(x) + a8 * _H8(x) + a12 * _H12(x))
    return np.abs(p * p * np.exp(-x * x))  # |g|^2, unnormalized


def _A_quad(a0, a4, a8):
    d = lambda x: _density(a0, a4, a8, x)
    norm, _ = quad(d, -np.inf, np.inf, limit=500)
    if norm <= 0 or not np.isfinite(norm):
        return np.inf
    num, _ = quad(lambda x: np.abs(x) * d(x), -np.inf, np.inf, limit=500)
    return num / norm


def _A_simp(a0, a4, a8):
    N = 8192
    x = np.linspace(-45.0, 45.0, N)
    d = _density(a0, a4, a8, x)
    norm = np.trapz(d, x)
    if norm <= 0:
        return np.inf
    return np.trapz(np.abs(x) * d, x) / norm


def _objective(v):
    a0, a4, a8 = v
    try:
        return _A_quad(a0, a4, a8)
    except Exception:
        return np.inf


def construct():
    global _T0
    best = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(best)

    def elapsed():
        return time.time() - _T0

    def time_left():
        return _DEADLINE - elapsed()

    # ---- global multi-start over a dense grid ----
    lo, hi = -1.0, 1.0
    npts = 11
    grid = np.linspace(lo, hi, npts)
    starts = []
    for a0 in grid:
        for a4 in grid:
            for a8 in grid:
                starts.append((a0, a4, a8))
    # add prior-art and a few heuristic starts
    starts.append(tuple(best.tolist()))
    starts.append((0.0, 0.0, 0.0))

    seen = set()
    for v0 in starts:
        if time_left() < 12.0:
            break
        key = (round(v0[0], 4), round(v0[1], 4), round(v0[2], 4))
        if key in seen:
            continue
        seen.add(key)
        try:
            res = minimize(_objective, np.array(v0), method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-14, "maxls": 50})
            if res.success or res.fun < best_val:
                cand = np.clip(res.x, -3.0, 3.0)
                cv = _objective(cand)
                if cv < best_val:
                    best_val = cv
                    best = cand
        except Exception:
            pass

    # ---- basin-hopping refinement from the current best ----
    for _ in range(6):
        if time_left() < 15.0:
            break
        try:
            bh = basinhopping(_objective, best, niter=12, T=0.25,
                              stepsize=0.35, minimizer_kwargs={"method": "L-BFGS-B",
                                                               "options": {"maxiter": 100, "ftol": 1e-14}})
            cand = np.clip(bh.x, -3.0, 3.0)
            cv = _objective(cand)
            if cv < best_val:
                best_val = cv
                best = cand
        except Exception:
            break

    # ---- high-precision final polish ----
    if time_left() > 5.0:
        try:
            res = minimize(_objective, best, method="L-BFGS-B",
                           options={"maxiter": 400, "ftol": 1e-16, "gtol": 1e-16, "maxls": 80})
            cand = np.clip(res.x, -3.0, 3.0)
            cv = _objective(cand)
            if cv < best_val:
                best_val = cv
                best = cand
        except Exception:
            pass

    # cross-check with an independent quadrature (do not trust if wildly off)
    try:
        alt = _A_simp(*best)
        if np.isfinite(alt) and abs(alt - best_val) < 1e-4:
            best_val = min(best_val, alt)
    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
