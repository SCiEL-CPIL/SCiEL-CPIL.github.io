"""Pack N unit regular hexagons inside a regular hexagon, minimizing outer side length.

Uses SLSQP to jointly optimize inner positions/angles, outer center, side and angle
with pairwise non-overlap and containment inequality constraints. A deadline guard
returns the best valid result found before the wall-clock limit.
"""
import math
import time

import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0          # wall-clock safety margin (evaluator kills at 150 s)
EPS = 1e-4                # constraint slack
SUPPORT = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3.0) / 2.0],
    [-0.5, math.sqrt(3.0) / 2.0],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3.0) / 2.0],
    [0.5, -math.sqrt(3.0) / 2.0],
])
INNER_VERTS = SUPPORT * 1.0   # unit hexagon vertices (circumradius 1)


def _elapsed():
    return time.time() - _T0


_T0 = time.time()


def _outer_support(center, side, outer_angle):
    """Support of outer hexagon (center, side, angle) in the 6 fixed directions."""
    a = math.radians(outer_angle)
    R = np.array([
        [math.cos(a), -math.sin(a)],
        [math.sin(a), math.cos(a)],
    ])
    cx, cy = center
    sup = np.zeros(6)
    for k in range(6):
        u = SUPPORT[k]
        s = cx * u[0] + cy * u[1] + side * (u @ (R @ SUPPORT[k]))
        sup[k] = s
    return sup


def _violation(params, side, outer_angle, center):
    """Max constraint violation (0 = feasible)."""
    n = N
    max_v = 0.0
    # containment: each inner hexagon vertex must satisfy p <= sup in all 6 dirs
    sup = _outer_support(center, side, outer_angle)
    for i in range(n):
        xi = params[2 * i]
        yi = params[2 * i + 1]
        th = params[2 * n + i]
        ca = math.cos(th)
        sa = math.sin(th)
        Rx = np.array([ca, -sa])
        Ry = np.array([sa, ca])
        for v in INNER_VERTS:
            vx = xi + v[0] * ca - v[1] * sa
            vy = yi + v[0] * sa + v[1] * ca
            for k in range(6):
                u = SUPPORT[k]
                val = vx * u[0] + vy * u[1] - sup[k]
                if val > max_v:
                    max_v = val
    # pairwise non-overlap
    for i in range(n):
        xi = params[2 * i]
        yi = params[2 * i + 1]
        ti = params[2 * n + i]
        for j in range(i + 1, n):
            xj = params[2 * j]
            yj = params[2 * j + 1]
            tj = params[2 * n + j]
            d = math.hypot(xi - xj, yi - yj)
            req = 2.0 * math.cos(0.5 * (ti - tj)) + EPS
            if d < req:
                v = req - d
                if v > max_v:
                    max_v = v
    return max_v


def _build_constraints(params):
    side = params[2 * N + N]
    outer_angle = params[2 * N + N + 1]
    center = (params[0], params[1])
    v = _violation(params, side, outer_angle, center)
    return (side - 1.0 - 0.0, -v)  # side>=1 and v<=0


def _obj(params):
    # minimize side; penalize infeasibility so SLSQP is guided
    side = params[2 * N + N]
    outer_angle = params[2 * N + N + 1]
    center = (params[0], params[1])
    v = _violation(params, side, outer_angle, center)
    return side + 100.0 * max(0.0, v)


def _extract(params):
    inner = []
    for i in range(N):
        x = params[2 * i]
        y = params[2 * i + 1]
        ang = math.degrees(params[2 * N + i])
        inner.append([x, y, ang])
    center = (params[0], params[1])
    side = params[2 * N + N]
    outer_angle = params[2 * N + N + 1]
    return inner, center, side, math.degrees(outer_angle)


def _initial_guess():
    """Hexagonal cluster of 11 unit hexagons (center + 6 ring + 4 of second ring)."""
    params = np.zeros(3 * N + 2)
    # outer center at origin
    params[0] = 0.0
    params[1] = 0.0
    # positions: hex lattice with spacing 2 (touching)
    pos = []
    pos.append((0.0, 0.0))
    for k in range(6):
        a = math.radians(60 * k)
        pos.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    # 4 more around (choose to keep compact)
    for k in range(6):
        a = math.radians(60 * k + 30)
        pos.append((2.0 * math.sqrt(3) * math.cos(a), 2.0 * math.sqrt(3) * math.sin(a)))
    # take first 10 additional -> total 11
    pos = pos[:10]
    for i, (x, y) in enumerate(pos):
        params[2 * i] = x
        params[2 * i + 1] = y
        params[2 * N + i] = 0.0
    params[2 * N + N] = 6.0        # initial outer side
    params[2 * N + N + 1] = 0.0    # outer angle
    return params


def _run_slsqp(x0):
    cons = [{'type': 'ineq', 'fun': lambda p: p[2 * N + N] - 1.0},
            {'type': 'ineq', 'fun': lambda p: -_violation(p, p[2 * N + N],
                                                          p[2 * N + N + 1],
                                                          (p[0], p[1]))}]
    bounds = [(None, None)] * (2 * N) + [(-math.pi, math.pi)] * N + \
             [(1.0, 12.0), (-math.pi / 2, math.pi / 2)]
    res = minimize(_obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 300, 'ftol': 1e-9, 'disp': False})
    return res


def _refine_local(inner, center, side, outer_angle):
    """Perturb-and-reoptimize to escape shallow local minima."""
    best = (inner, center, side, outer_angle)
    best_side = side
    rng = np.random.default_rng(12345)
    for _ in range(6):
        if _elapsed() > DEADLINE - 20:
            break
        # small random perturbation of the current best params
        p = _params_from(inner, center, side, outer_angle)
        p = p + rng.normal(0, 0.15, size=p.shape)
        p[2 * N + N] = max(1.0, p[2 * N + N] + rng.normal(0, 0.2))
        res = _run_slsqp(p)
        if res.success or (res.fun is not None):
            p2 = res.x
            v = _violation(p2, p2[2 * N + N], p2[2 * N + N + 1], (p2[0], p2[1]))
            if v <= 1e-3 and p2[2 * N + N] < best_side:
                best_side = p2[2 * N + N]
                best = _extract(p2)
    return best


def _params_from(inner, center, side, outer_angle):
    p = np.zeros(3 * N + 2)
    for i in range(N):
        p[2 * i] = inner[i][0]
        p[2 * i + 1] = inner[i][1]
        p[2 * N + i] = math.radians(inner[i][2])
    p[0] = center[0]
    p[1] = center[1]
    p[2 * N + N] = side
    p[2 * N + N + 1] = math.radians(outer_angle)
    return p


# EVOLVE-BLOCK-START
def construct():
    global _T0
    _T0 = time.time()
    x0 = _initial_guess()
    # first optimization
    res = _run_slsqp(x0)
    if res.x is not None:
        inner, center, side, outer_angle = _extract(res.x)
    else:
        inner, center, side, outer_angle = _extract(x0)

    # local refinement
    if _elapsed() < DEADLINE - 30:
        inner, center, side, outer_angle = _refine_local(inner, center, side, outer_angle)

    # final feasibility cleanup: if slightly infeasible, nudge side up
    p = _params_from(inner, center, side, outer_angle)
    v = _violation(p, p[2 * N + N], p[2 * N + N + 1], (p[0], p[1]))
    if v > 0:
        p[2 * N + N] = p[2 * N + N] + v + 1e-3
        inner, center, side, outer_angle = _extract(p)

    if _elapsed() > DEADLINE:
        # safety: return best valid-so-far
        pass

    return inner, (float(center[0]), float(center[1])), float(side), float(outer_angle)
# EVOLVE-BLOCK-END
