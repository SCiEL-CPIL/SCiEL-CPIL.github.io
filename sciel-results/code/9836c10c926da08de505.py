"""Packing 12 unit hexagons in a minimal regular hexagon via SAT + Nelder-Mead."""
import time
import math
import numpy as np

N = 12
DEADLINE = 145.0  # safety margin under 150s wall-clock

# Precompute unit hexagon vertices (flat-top, side=1)
def _unit_hex_vertices():
    verts = []
    for k in range(6):
        a = math.radians(60.0 * k)
        verts.append((math.cos(a), math.sin(a)))
    return np.array(verts)

UNIT_VERTS = _unit_hex_vertices()

def _hex_vertices(cx, cy, angle_deg, scale=1.0):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return np.array([[cx + scale * (R @ v)[0], cy + scale * (R @ v)[1]] for v in UNIT_VERTS])

def _edge_normals(verts):
    n = len(verts)
    norms = np.zeros((n, 2))
    for i in range(n):
        p0 = verts[i]
        p1 = verts[(i + 1) % n]
        e = p1 - p0
        L = math.hypot(e[0], e[1])
        if L < 1e-12:
            norms[i] = [1.0, 0.0]
        else:
            norms[i] = [-e[1] / L, e[0] / L]
    return norms

def _sat_overlap(vA, nA, vB, nB):
    """Return penetration depth (positive if overlapping, 0 if separated) via SAT on combined axes."""
    min_overlap = float('inf')
    for axis in list(nA) + list(nB):
        projA = vA @ axis
        projB = vB @ axis
        minA, maxA = projA.min(), projA.max()
        minB, maxB = projB.min(), projB.max()
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap <= 0.0:
            return 0.0
        if overlap < min_overlap:
            min_overlap = overlap
    return min_overlap

def _outer_containment(vInner, vOuter):
    """Penetration depth of inner hexagon beyond outer (positive if inner pokes out)."""
    # Axes: outer edge normals (inward containment) — inner must be within
    min_pen = 0.0
    nO = _edge_normals(vOuter)
    # For containment, project both; inner's extent must be within outer's extent on each axis
    for axis in nO:
        projI = vInner @ axis
        projO = vOuter @ axis
        minI, maxI = projI.min(), projI.max()
        minO, maxO = projO.min(), projO.max()
        # inner must fit: minI >= minO and maxI <= maxO
        pen = 0.0
        if minI < minO:
            pen = minO - minI
        if maxI > maxO:
            pen = max(pen, maxI - maxO)
        if pen > min_pen:
            min_pen = pen
    return min_pen

def _decode(params):
    # params: for each inner: x, y, angle(deg); then outer: cx, cy, side, angle(deg)
    inner = []
    for i in range(N):
        x = params[3 * i]
        y = params[3 * i + 1]
        a = params[3 * i + 2]
        inner.append((x, y, a))
    ocx = params[3 * N]
    ocy = params[3 * N + 1]
    side = params[3 * N + 2]
    oang = params[3 * N + 3]
    return inner, (ocx, ocy), side, oang

def _objective(params):
    inner, (ocx, ocy), side, oang = _decode(params)
    vOuter = _hex_vertices(ocx, ocy, oang, scale=side)
    pen = 0.0
    inner_verts = []
    for (x, y, a) in inner:
        v = _hex_vertices(x, y, a, scale=1.0)
        inner_verts.append(v)
        # containment
        cpen = _outer_containment(v, vOuter)
        if cpen > pen:
            pen = cpen
    # pairwise
    for i in range(N):
        for j in range(i + 1, N):
            vi = inner_verts[i]
            vj = inner_verts[j]
            ni = _edge_normals(vi)
            nj = _edge_normals(vj)
            p = _sat_overlap(vi, ni, vj, nj)
            if p > pen:
                pen = p
    # side penalty (encourage small side)
    return pen * 1000.0 + side

def _initial_params():
    # Place hexagons in a rough grid, outer large enough to be valid
    rng = np.random.default_rng(0)
    params = np.zeros(3 * N + 4)
    # hex grid positions for 12 hexagons (side 1, spacing ~2.0 to avoid overlap)
    positions = []
    # ring arrangement: 1 center + 6 ring + 5 more
    positions.append((0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k)
        positions.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    # 5 more on outer ring
    for k in range(5):
        a = math.radians(60.0 * k + 30.0)
        positions.append((3.8 * math.cos(a), 3.8 * math.sin(a)))
    for i in range(N):
        params[3 * i] = positions[i][0]
        params[3 * i + 1] = positions[i][1]
        params[3 * i + 2] = 0.0
    params[3 * N] = 0.0
    params[3 * N + 1] = 0.0
    params[3 * N + 2] = 12.0  # start with large outer
    params[3 * N + 3] = 0.0
    return params

def _nelder_mead(f, x0, max_iter, tol=1e-8, time_limit=None, start_time=None):
    from scipy.optimize import minimize
    if start_time is None:
        start_time = time.time()
    def _wrapped(x):
        val = f(x)
        if time_limit is not None and time.time() - start_time > time_limit:
            raise TimeoutError()
        return val
    try:
        res = minimize(_wrapped, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-5,
                                'adaptive': True})
        return res.x, res.fun
    except TimeoutError:
        return x0, f(x0)
    except Exception:
        return x0, f(x0)

def construct():
    start = time.time()
    best_params = _initial_params()
    best_val = _objective(best_params)

    # Try a few restarts with different initial spreads
    rng = np.random.default_rng(12345)
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() - start > DEADLINE - 5:
            break
        x0 = best_params.copy()
        if r > 0:
            # perturb
            x0[0:3*N] += rng.normal(0, 0.5, size=3*N)
            x0[3*N+2] = 10.0 + rng.uniform(0, 4)
            x0[3*N+3] = rng.uniform(0, 60)
        # time budget per restart
        remaining = DEADLINE - 5 - (time.time() - start)
        if remaining < 3:
            break
        max_iter = int(min(400, max(50, remaining * 15)))
        x, val = _nelder_mead(_objective, x0, max_iter, time_limit=remaining - 2, start_time=start)
        if val < best_val:
            best_val = val
            best_params = x

    inner, (ocx, ocy), side, oang = _decode(best_params)
    # Ensure side is at least a sane minimum
    side = max(side, 2.0)
    return inner, (ocx, ocy), side, oang
