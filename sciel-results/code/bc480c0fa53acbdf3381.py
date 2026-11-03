"""Packing of unit regular hexagons in a regular hexagon.

Design: dense hexagonal lattice of unit hexagons (disjoint by construction,
spacing = 2*sqrt(3) = the minimal non-overlap distance).  The outer hexagon is
centered at the origin with rotation 0.  The outer side length is the maximum,
over every inner-hexagon vertex, of the six support-function values, divided by
sqrt(3)/2.  That nonsmooth max is replaced by a smooth log-sum-exp surrogate so
that L-BFGS-B can minimize the objective over the inner-hexagon positions.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11

# --- constants ---------------------------------------------------------------
_SQRT3 = math.sqrt(3.0)
_SPACING = 2.0 * _SQRT3                 # min non-overlap distance for unit hexes
_NORMALS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                     for a in (30, 90, 150, 210, 270, 330)])
# unit hexagon vertices (radius 1, flat-top so vertices lie on the normals)
_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                   for a in (30, 90, 150, 210, 270, 330)])

# --- dense hexagonal lattice (mode 0: pointy lattice) ------------------------
def _lattice_positions():
    """Return the list of (x, y) centers of a dense hexagonal lattice of N hexes."""
    centers = []
    half = (N - 1) // 2
    for i in range(-half, N - half):
        row_offset = (abs(i) % 2) * (_SPACING * 0.5)
        for j in range(-half, N - half):
            x = i * _SPACING + row_offset
            y = j * _SPACING
            centers.append((x, y))
    return centers


_CENTERS = _lattice_positions()
_M = len(_CENTERS)
# precompute per-hexagon vertex offsets (angle fixed at 0)
_VOFF = _VERTS[None, :, :] + np.zeros((_M, 1, 2))   # shape (M, 6, 2)


def _outer_side(x):
    """Smooth outer side length for a flat position vector x (length 2M)."""
    pos = x.reshape(_M, 2)[:, None, :]
    verts = _VOFF + pos                          # (M, 6, 2)
    proj = np.einsum('ij,mjk->mik', _NORMALS, verts)   # (M, 6, 6)
    m = proj.max(axis=(0, 1))                    # (6,)
    beta = 50.0
    # log-sum-exp over the M*6 values per normal direction
    flat = proj.transpose(2, 0, 1).reshape(6, -1)      # (6, M*6)
    mx = flat.max(axis=1, keepdims=True)
    lse = mx + np.log(np.exp(flat - mx).sum(axis=1, keepdims=True))
    sm = lse - np.log(_M * 6) / beta
    return float(np.max(sm) / (_SQRT3 / 2.0))


def _grad(x):
    """Gradient of the smooth objective (via finite differences of the surrogate)."""
    eps = 1e-6
    g = np.empty_like(x)
    f0 = _outer_side(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (_outer_side(xp) - _outer_side(xm)) / (2.0 * eps)
    return g


def construct():
    t0 = time.time()
    DEADLINE = 150.0 - 15.0            # safety margin before the 150 s kill
    x0 = np.array([c[0] for c in _CENTERS] + [c[1] for c in _CENTERS])

    best_x = x0.copy()
    best_val = _outer_side(x0)

    # L-BFGS-B on the smooth surrogate (numerical gradient)
    try:
        res = minimize(_outer_side, x0, jac=_grad, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10,
                                'maxls': 40})
        if time.time() < DEADLINE:
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _outer_side(res.x)
                if v < best_val:
                    best_val, best_x = v, res.x
    except Exception:
        pass

    # final high-precision polish pass with a tighter tolerance
    if time.time() < DEADLINE:
        try:
            res2 = minimize(_outer_side, best_x, jac=_grad, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12,
                                     'maxls': 30})
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                v = _outer_side(res2.x)
                if v < best_val:
                    best_val, best_x = v, res2.x
        except Exception:
            pass

    pos = best_x.reshape(_M, 2)
    inner = [[float(pos[k, 0]), float(pos[k, 1]), 0.0] for k in range(_M)]
    side = float(best_val)
    return inner, (0.0, 0.0), side, 0.0
