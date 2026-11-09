"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0
_SAFETY = 40.0

def _deadline_reached():
    return (time.time() - _START) >= (_DEADLINE - _SAFETY)

# Precompute Laguerre basis matrix B[k, j] = coeff of x^j in L_k(x)
_ORDER = 24
_B = np.zeros((_ORDER + 1, _ORDER + 1))
for _k in range(_ORDER + 1):
    for _j in range(_k + 1):
        _B[_k, _j] = ((-1.0) ** _j) * (math.comb(_k, _j) / math.factorial(_j))
_BT_INV = None
def _basis_inv():
    global _BT_INV
    if _BT_INV is None:
        try:
            _BT_INV = np.linalg.inv(_B.T)
        except np.linalg.LinAlgError:
            _BT_INV = np.linalg.pinv(_B.T)
    return _BT_INV

def _laguerre_coeffs(poly_coeffs):
    d = len(poly_coeffs) - 1
    c = np.zeros(_ORDER + 1)
    for idx, coef in enumerate(poly_coeffs):
        power = d - idx
        if 0 <= power <= _ORDER:
            c[power] = coef
    return _basis_inv() @ c

def _poly_from_roots(roots):
    p = np.array([1.0])
    for r in roots:
        p = np.convolve(p, np.array([1.0, -2.0 * r, r * r]))
    return p

def _c4(roots):
    return 4.0 * float(np.max(roots))

def _penalty(roots):
    a = _laguerre_coeffs(_poly_from_roots(roots))
    neg = np.maximum(0.0, -a)
    return float(np.sum(neg * neg))

def _objective(x):
    roots = np.maximum(x, 1e-6)
    return _c4(roots) + 200.0 * _penalty(roots)

def construct():
    best_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    best_val = _objective(best_roots)

    bounds = [(1e-3, 60.0)] * 12
    if not _deadline_reached():
        try:
            res = differential_evolution(
                _objective, bounds, maxiter=120, tol=1e-5,
                seed=1, polish=True, popsize=12, init='sobol',
            )
            if res.x is not None:
                cand = np.maximum(res.x, 1e-6)
                cv = _objective(cand)
                if cv < best_val:
                    best_roots, best_val = cand, cv
        except Exception:
            pass

    if not _deadline_reached():
        try:
            res2 = minimize(
                _objective, best_roots, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12},
            )
            if res2.x is not None:
                cand = np.maximum(res2.x, 1e-6)
                cv = _objective(cand)
                if cv < best_val:
                    best_roots, best_val = cand, cv
        except Exception:
            pass

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
