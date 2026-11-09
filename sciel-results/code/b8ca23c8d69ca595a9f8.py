# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 630.0
_START = None


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _separation(v1, v2):
    """Return the minimum separation between two convex hexagons.
    Positive => disjoint, negative => overlap (magnitude ~ penetration).
    Uses all 12 edge normals (6 from each)."""
    e1 = np.roll(v1, -1, axis=0) - v1
    e2 = np.roll(v2, -1, axis=0) - v2
    n1 = np.column_stack([e1[:, 1], -e1[:, 0]])
    n2 = np.column_stack([e2[:, 1], -e2[:, 0]])
    best = np.inf
    for N in (n1, n2):
        for i in range(6):
            n = N[i]
            p1 = v1.dot(n)
            p2 = v2.dot(n)
            a1, b1 = p1.min(), p1.max()
            a2, b2 = p2.min(), p2.max()
            if b2 < a1:
                sep = a1 - b2
            elif b1 < a2:
                sep = a2 - b1
            else:
                sep = 0.0
            if sep < best:
                best = sep
    return best


def _outside_penalty(verts, outer_v):
    """Sum of squared negative cross products for each vertex against each
    outer edge (p1 -> p2). cross(p2-p1, p-p1) >= 0 means inside."""
    e = np.roll(outer_v, -1, axis=0) - outer_v
    # e[i] is edge from outer_v[i] to outer_v[i+1]
    total = 0.0
    for i in range(6):
        p1 = outer_v[i]
        ed = e[i]
        cross = ed[0] * (verts[:, 1] - p1[1]) - ed[1] * (verts[:, 0] - p1[0])
        neg = np.minimum(cross, 0.0)
        total += float(np.sum(neg * neg))
    return total


def _evaluate(x, S):
    """x: (11,3) = (cx, cy, a_deg). Returns (penalty, side_penalty, min_sep)."""
    outer_v = _hex_vertices(0.0, 0.0, S, 0.0)
    verts_list = [_hex_vertices(x[i, 0], x[i, 1], 1.0, x[i, 2]) for i in range(11)]
    min_sep = np.inf
    sep_pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            sep = _separation(verts_list[i], verts_list[j])
            if sep < min_sep:
                min_sep = sep
            if sep < 1e-6:
                sep_pen += (1e-6 - sep) ** 2
    out_pen = 0.0
    for i in range(11):
        out_pen += _outside_penalty(verts_list[i], outer_v)
    return sep_pen, out_pen, min_sep


def _objective(x, S, lam):
    sep_pen, out_pen, _ = _evaluate(x, S)
    return S + lam * (sep_pen + out_pen)


def _is_valid(x, S):
    sep_pen, out_pen, min_sep = _evaluate(x, S)
    if not np.all(np.isfinite(x)):
        return False
    if S < np.sqrt(11.0) - 1e-9:
        return False
    if min_sep <= 1e-6:
        return False
    if out_pen > 1e-12:
        return False
    # verify edge lengths within 1e-8 of 1
    for i in range(11):
        v = _hex_vertices(x[i, 0], x[i, 1], 1.0, x[i, 2])
        e = np.roll(v, -1, axis=0) - v
        L = np.linalg.norm(e, axis=1)
        if np.max(np.abs(L - 1.0)) > 1e-8:
            return False
    return True


def _refine(x, S, lam, deadline):
    """Local refinement using L-BFGS-B, then Nelder-Mead polish."""
    global _START
    best = x.copy()
    best_val = _objective(x, S, lam)
    try:
        res = minimize(
            lambda v: _objective(v, S, lam),
            x,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            val = _objective(res.x, S, lam)
            if val < best_val:
                best = res.x
                best_val = val
        if time.time() - _START > deadline:
            return best
        # polish
        res2 = minimize(
            lambda v: _objective(v, S, lam),
            best,
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-9, "fatol": 1e-12, "adaptive": True},
        )
        if res2.x is not None and np.all(np.isfinite(res2.x)):
            val = _objective(res2.x, S, lam)
            if val < best_val:
                best = res2.x
                best_val = val
    except Exception:
        pass
    return best


def _initial_configs():
    """A set of diverse starting points."""
    configs = []

    # 1) Hexagonal lattice cluster (7) + 4 extras
    r = 2.0
    h = r * np.sqrt(3) / 2
    lattice = [
        (0, 0),
        (r, 0), (-r, 0),
        (r/2, h), (-r/2, h),
        (r/2, -h), (-r/2, -h),
    ]
    extra = [
        (r*1.5, h),
        (r*1.5, -h),
        (-r*1.5, h),
        (-r*1.5, -h),
    ]
    pts = lattice + extra
    x = np.array([[p[0], p[1], 0.0] for p in pts])
    configs.append(x)

    # 2) Ring of 6 around center + 4 more
    ang = np.deg2rad(60 * np.arange(6))
    ring = [(1.9 * np.cos(a), 1.9 * np.sin(a)) for a in ang]
    center = [(0.0, 0.0)]
    outer2 = [(3.2 * np.cos(a), 3.2 * np.sin(a)) for a in np.deg2rad(30 + 60 * np.arange(4))]
    pts = center + ring + outer2
    x = np.array([[p[0], p[1], 0.0] for p in pts])
    configs.append(x)

    # 3) Random
    rng = np.random.default_rng(12345)
    for _ in range(3):
        cx = rng.uniform(-3, 3, 11)
        cy = rng.uniform(-3, 3, 11)
        a = rng.uniform(-30, 30, 11)
        x = np.column_stack([cx, cy, a])
        configs.append(x)

    return configs


def hexagon_packing_11():
    global _START, _DEADLINE
    _START = time.time()
    deadline = _DEADLINE - 20.0  # safety margin

    # Start with a generous S, then anneal down.
    S = 8.0
    lam = 50.0

    best_x = None
    best_val = np.inf

    configs = _initial_configs()

    # Annealing loop: shrink S and increase penalty weight.
    schedule = [
        (8.0, 50.0),
        (6.5, 80.0),
        (5.5, 120.0),
        (5.0, 180.0),
        (4.7, 250.0),
        (4.5, 350.0),
        (4.3, 500.0),
        (4.2, 700.0),
        (4.1, 1000.0),
    ]

    for S_target, lam in schedule:
        if time.time() - _START > deadline:
            break
        # Move S toward target
        S = S_target
        # Refine from each initial config (and from current best)
        starts = configs[:]
        if best_x is not None:
            starts.append(best_x.copy())
        for x0 in starts:
            if time.time() - _START > deadline:
                break
            x1 = _refine(x0, S, lam, deadline)
            val = _objective(x1, S, lam)
            if val < best_val:
                best_val = val
                best_x = x1
        configs = [best_x.copy()] if best_x is not None else configs

    # Final polish with a very high penalty to enforce validity.
    if best_x is not None:
        for lam in (1500.0, 3000.0):
            if time.time() - _START > deadline:
                break
            x1 = _refine(best_x, S, lam, deadline)
            if _objective(x1, S, lam) <= _objective(best_x, S, lam) + 1e-9:
                best_x = x1

    # Now find the smallest S that keeps best_x valid by binary search.
    if best_x is not None:
        lo = np.sqrt(11.0) + 1e-6
        hi = S
        # ensure hi is valid
        if not _is_valid(best_x, hi):
            # expand hi
            hi = 10.0
            while not _is_valid(best_x, hi) and hi < 30.0:
                hi *= 1.2
        for _ in range(30):
            if time.time() - _START > deadline:
                break
            mid = 0.5 * (lo + hi)
            if _is_valid(best_x, mid):
                hi = mid
            else:
                lo = mid
        S_final = hi
    else:
        # Fallback: the seed grid (should be valid)
        best_x = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ],
            dtype=float,
        )
        S_final = 8.0

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return best_x, outer_hex_data, float(S_final)


# EVOLVE-BLOCK-END
