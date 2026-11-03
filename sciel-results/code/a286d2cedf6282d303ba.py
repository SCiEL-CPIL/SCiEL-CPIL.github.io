"""Rewritten program: numerically optimize the Hermite coefficients to reduce
the uncertainty-inequality upper bound (a new approach family vs. the seed)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Mathematical setup (faithful reconstruction of the notebook's quantities)
# ---------------------------------------------------------------------------

def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via the three-term recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    Hnm1 = 2.0 * x
    Hn = 4.0 * x * x - 2.0
    for k in range(2, n + 1):
        Hnm2 = Hnm1
        Hnm1 = Hn
        Hn = 2.0 * k * Hnm1 - (2.0 * k - 1.0) * Hnm2
    return Hn


def _build_grid():
    """Symmetric grid and precomputed Hermite polynomials."""
    x = np.linspace(-12.0, 12.0, 4001)
    H = {}
    for n in (0, 4, 8, 12):
        H[n] = _hermite(n, x)
    return x, H


def _gauss(x):
    return np.exp(-0.5 * x * x)


# Precompute once (module level) so each construct() call is cheap.
_X, _H = _build_grid()
_G = _gauss(_X)
_DX = _X[1] - _X[0]
_H0 = _H[0]
_H4 = _H[4]
_H8 = _H[8]
_H12 = _H[12]


def _coeffs_from_frees(a):
    """Given free coefficients (a0,a4,a8), derive a12 so that g(0)=0."""
    a0, a4, a8 = a
    # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * g(0)
    # H_n(0) = 0 for odd n, = (-1)^(n/2)*(n-1)!! for even n.
    H0_0 = 1.0
    H4_0 = -3.0
    H8_0 = 105.0
    H12_0 = -10395.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    return a0, a4, a8, a12


def _norms_from_frees(a):
    """Return (L1, Linf, L2sq) of g for the free coefficients a."""
    a0, a4, a8, a12 = _coeffs_from_frees(a)
    g = (a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12) * _G
    L1 = np.sum(np.abs(g)) * _DX
    Linf = np.max(np.abs(g))
    L2sq = np.sum(g * g) * _DX
    return L1, Linf, L2sq


def _bound_from_frees(a):
    """Upper bound on C4 = A(f)*A(Fourier(f)) for the test function."""
    L1, Linf, L2sq = _norms_from_frees(a)
    if L1 <= 0 or L2sq <= 0:
        return np.inf
    A = L1 / np.sqrt(L2sq)          # L1/L2 norm
    B = Linf / np.sqrt(L2sq)        # Linf/L2 norm
    return A * B


def _objective(a):
    try:
        v = _bound_from_frees(a)
    except Exception:
        return 1e12
    if not np.isfinite(v):
        return 1e12
    return float(v)


# ---------------------------------------------------------------------------
# Reference (prior state of the art) as a starting point / fallback
# ---------------------------------------------------------------------------
_REF = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _optimize():
    """Run a local optimizer from several starts; return the best coefficients."""
    starts = [
        _REF.copy(),
        _REF * 0.5,
        _REF * 2.0,
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.04, 0.0005]),
        np.array([-0.9, 0.06, 0.0002]),
        np.zeros(3),
        np.array([-1.0, 0.0, 0.0]),
    ]

    best = _REF.copy()
    best_val = _objective(_REF)

    for s in starts:
        try:
            res = minimize(
                _objective,
                s,
                method="Nelder-Mead",
                options={
                    "maxiter": 4000,
                    "maxfev": 6000,
                    "xatol": 1e-9,
                    "fatol": 1e-12,
                    "adaptive": True,
                },
            )
            val = res.fun
            if np.isfinite(val) and val < best_val:
                best_val = val
                best = np.array(res.x, dtype=float)
        except Exception:
            continue

    # Final polish with a tighter run from the best found point.
    try:
        res = minimize(
            _objective,
            best,
            method="Nelder-Mead",
            options={
                "maxiter": 8000,
                "maxfev": 12000,
                "xatol": 1e-12,
                "fatol": 1e-14,
                "adaptive": True,
            },
        )
        if np.isfinite(res.fun) and res.fun < best_val:
            best = np.array(res.x, dtype=float)
    except Exception:
        pass

    # Safety: never return something that blows up the bound.
    if not np.isfinite(_objective(best)) or _objective(best) > best_val:
        best = _REF
    return best


def construct():
    """Return the three free Hermite coefficients (degrees 0, 4, 8)."""
    try:
        return _optimize()
    except Exception:
        return _REF.copy()
# EVOLVE-BLOCK-END
