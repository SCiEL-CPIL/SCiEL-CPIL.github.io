import numpy as np
from scipy.optimize import minimize
import time

N = 12

# Hex lattice basis: unit hexagons (side 1) with a vertex on +x axis tile with
# these two edge vectors; adjacent cells share an edge => guaranteed disjoint.
E1 = np.array([1.5, np.sqrt(3) / 2])
E2 = np.array([0.0, np.sqrt(3)])


def _build_block():
    """4x3 rigid block of aligned unit hexagons (all orientation 0 deg)."""
    pts = []
    for i in range(4):
        for j in range(3):
            c = i * E1 + j * E2
            pts.append((c[0], c[1], 0.0))
    return pts


def _req_side(cx, cy, phi, centers, thetas):
    """Minimum outer side length for given outer center (cx,cy) and orientation phi (rad)."""
    js = np.arange(6)
    best = 0.0
    for j in range(6):
        b = phi + j * np.pi / 3
        nx, ny = np.cos(b), np.sin(b)
        maxval = -1e18
        for k in range(len(centers)):
            qx, qy = centers[k]
            # support function of unit hexagon (vertices at thetas[k]+k*60deg) in direction b
            va = thetas[k] + js * np.pi / 3
            sup = np.cos(b - va).max()
            val = qx * nx + qy * ny + sup
            if val > maxval:
                maxval = val
        rhs = maxval - (cx * nx + cy * ny)
        if rhs > best:
            best = rhs
    # apothem = s*sqrt(3)/2  =>  s = (2/sqrt(3)) * apothem
    return (2.0 / np.sqrt(3)) * best


def construct():
    t0 = time.time()
    deadline = t0 + 118.0  # explicit guard, safety margin before 150s kill

    block = _build_block()
    centers0 = np.array([(p[0], p[1]) for p in block])
    thetas = np.array([p[2] for p in block])
    com = centers0.mean(axis=0)
    centers = centers0 - com  # work in centered frame; add com back at the end

    def obj(v):
        return _req_side(v[0], v[1], np.deg2rad(v[2]), centers, thetas)

    best = np.array([0.0, 0.0, 0.0])
    best_val = obj(best)

    # Multi-start over outer rotation (and center) to avoid local maxima of the max-of-6.
    starts = [(0.0, 0.0, 0.0), (0.0, 0.0, 30.0), (0.0, 0.0, 90.0),
              (0.0, 0.0, 15.0), (0.0, 0.0, 60.0), (0.0, 0.0, 45.0),
              (0.0, 0.0, 75.0), (0.0, 0.0, 105.0)]
    for s in starts:
        if time.time() > deadline - 8.0:
            break
        try:
            res = minimize(obj, s, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-7,
                                    'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_val:
                best_val = float(res.fun)
                best = np.array(res.x, dtype=float)
        except Exception:
            pass

    # High-precision polish pass (tighter tolerance) on the best candidate.
    if time.time() < deadline - 3.0:
        try:
            res = minimize(obj, best, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-9,
                                    'fatol': 1e-11, 'adaptive': True})
            if res.fun < best_val:
                best_val = float(res.fun)
                best = np.array(res.x, dtype=float)
        except Exception:
            pass

    cx, cy, phi_deg = best
    inner = [[float(c[0] + com[0]), float(c[1] + com[1]), float(t)]
             for c, t in zip(centers, thetas)]
    outer_center = (float(com[0] + cx), float(com[1] + cy))
    side = float(best_val) * (1.0 + 1e-9)  # tiny margin so containment holds under fp
    return inner, outer_center, side, float(phi_deg)
