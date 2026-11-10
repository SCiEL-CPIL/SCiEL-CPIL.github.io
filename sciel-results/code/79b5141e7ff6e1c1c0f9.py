# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _all_inner_verts(inner, ox, oy):
    """Vertices (66,2) of all 11 inner hexagons, shifted by outer center."""
    a = np.radians(inner[:, 2])
    angles = a[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts = np.stack([
        inner[:, 0] - ox + np.cos(angles),
        inner[:, 1] - oy + np.sin(angles)
    ], axis=2)  # (11,6,2)
    return verts.reshape(66, 2)


def _outer_verts(side, oa_deg, ox, oy):
    a = np.radians(oa_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([ox + side * np.cos(angles), oy + side * np.sin(angles)])


def _edge_normals(verts):
    """Unit outward normals of 6 edges of one hexagon. verts (6,2)."""
    e = np.roll(verts, -1, axis=0) - verts
    n = np.column_stack([e[:, 1], -e[:, 0]])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return n


def _separation_violation(inner, ox, oy):
    """Return max overlap (positive = violation) across all inner pairs. 0 if separated."""
    V = _all_inner_verts(inner, ox, oy)
    worst = 0.0
    for i in range(11):
        vi = V[6 * i: 6 * i + 6]
        ni = _edge_normals(vi)
        for j in range(i + 1, 11):
            vj = V[6 * j: 6 * j + 6]
            nj = _edge_normals(vj)
            for n in ni:
                pi = vi @ n
                pj = vj @ n
                ov = min(pi.max(), pj.max()) - max(pi.min(), pj.min())
                if ov > worst:
                    worst = ov
            for n in nj:
                pi = vi @ n
                pj = vj @ n
                ov = min(pi.max(), pj.max()) - max(pi.min(), pj.min())
                if ov > worst:
                    worst = ov
    return worst


def _containment_violation(inner, side, oa_deg, ox, oy):
    """Max distance (positive = violation) of inner vertices outside the outer hexagon."""
    V = _all_inner_verts(inner, ox, oy)
    OV = _outer_verts(side, oa_deg, ox, oy)
    worst = 0.0
    for k in range(6):
        p1 = OV[k]
        p2 = OV[(k + 1) % 6]
        edge = p2 - p1
        # cross(edge, V - p1) >= 0 for interior (CCW)
        cross = edge[0] * (V[:, 1] - p1[1]) - edge[1] * (V[:, 0] - p1[0])
        # distance outside = -cross / |edge|
        d = -cross / np.linalg.norm(edge)
        dmax = d.max()
        if dmax > worst:
            worst = dmax
    return worst


def _side_from_positions(inner, oa_deg, ox, oy):
    """Minimum outer side length to contain all inner hexagons (outer centered at origin)."""
    V = _all_inner_verts(inner, ox, oy)
    a = np.radians(oa_deg)
    normals = np.column_stack([
        np.cos(a + np.radians(30 + 60 * np.arange(6))),
        np.sin(a + np.radians(30 + 60 * np.arange(6)))
    ])
    max_proj = (V @ normals.T).max(axis=0)
    return float(np.max(max_proj) * 2.0 / np.sqrt(3.0))


def _check_validity(inner, outer, side, tol=1e-6):
    ox, oy, oa = outer
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        return False
    if not np.isfinite(side) or side < np.sqrt(11) - 1e-9:
        return False
    if _separation_violation(inner, ox, oy) > tol:
        return False
    if _containment_violation(inner, side, oa, ox, oy) > tol:
        return False
    return True


def _objective(params):
    """Minimize outer side length; penalize violations heavily."""
    x = params[0:11]
    y = params[11:22]
    a = params[22:33]
    oa = params[33]
    side = params[34]

    inner = np.column_stack([x, y, a])
    ox, oy = 0.0, 0.0

    sep = _separation_violation(inner, ox, oy)
    con = _containment_violation(inner, side, oa, ox, oy)

    # required side
    req_side = _side_from_positions(inner, oa, ox, oy)

    # Penalty: if separation violated, or side too small to contain, or side < required
    pen = 0.0
    if sep > 1e-6:
        pen += 100.0 * sep + 5.0
    if side < req_side - 1e-6:
        pen += 100.0 * (req_side - side) + 5.0
    if con > 1e-6:
        pen += 100.0 * con + 5.0

    return side + pen


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, side).
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 25.0

    sqrt3 = np.sqrt(3.0)

    # Flower pattern: 1 center + 6 at distance 2 + 4 corners at distance 3
    # (unit hexagons, flat-top, angle 0). Distance 2 = touching; we add a small gap.
    def make_initial(gap=0.0):
        d1 = 2.0 + gap
        d2 = 3.0 + gap
        pts = [
            [0.0, 0.0],
            [d1, 0.0],
            [d1 / 2, d1 * sqrt3 / 2],
            [-d1 / 2, d1 * sqrt3 / 2],
            [-d1, 0.0],
            [-d1 / 2, -d1 * sqrt3 / 2],
            [d1 / 2, -d1 * sqrt3 / 2],
            [1.5 * d1, 0.5 * d1 * sqrt3],
            [-1.5 * d1, 0.5 * d1 * sqrt3],
            [1.5 * d1, -0.5 * d1 * sqrt3],
            [-1.5 * d1, -0.5 * d1 * sqrt3],
        ]
        return np.array(pts, dtype=float)

    # Start with a guaranteed-separated config (small gap)
    inner0 = make_initial(gap=0.02)
    inner0 = np.column_stack([inner0[:, 0], inner0[:, 1], np.zeros(11)])

    # Best valid result so far
    best_inner = inner0.copy()
    best_oa = 0.0
    best_side = _side_from_positions(inner0, best_oa, 0.0, 0.0)
    best_valid = _check_validity(best_inner, np.array([0.0, 0.0, best_oa]), best_side)

    def try_update(inner, oa):
        nonlocal best_inner, best_oa, best_side, best_valid
        side = _side_from_positions(inner, oa, 0.0, 0.0)
        if _check_validity(inner, np.array([0.0, 0.0, oa]), side):
            if side < best_side - 1e-9:
                best_inner = inner.copy()
                best_oa = oa
                best_side = side
                best_valid = True
                return True
        return False

    # --- Phase 1: Nelder-Mead optimization over (x, y, angle, oa, side) ---
    def run_nm(x0, y0, a0, oa0, side0, maxiter, tol):
        params0 = np.concatenate([x0, y0, a0, [oa0, side0]])
        res = minimize(
            _objective, params0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol,
                     'adaptive': True, 'initial_simplex': 0.05}
        )
        p = res.x
        inner = np.column_stack([p[0:11], p[11:22], p[22:33]])
        oa = float(p[33])
        return inner, oa

    # Initial parameter vector
    x0 = inner0[:, 0].copy()
    y0 = inner0[:, 1].copy()
    a0 = np.zeros(11)
    oa0 = 0.0
    side0 = best_side

    # Run a few Nelder-Mead passes, tightening each time
    for k, (maxiter, tol) in enumerate([(400, 1e-5), (800, 1e-6), (1200, 1e-7)]):
        if time.time() > deadline - safety_margin:
            break
        inner, oa = run_nm(x0, y0, a0, oa0, side0, maxiter, tol)
        # Normalize angles
        inner[:, 2] = np.degrees(np.arctan2(np.sin(np.radians(inner[:, 2])),
                                             np.cos(np.radians(inner[:, 2]))))
        if try_update(inner, oa):
            x0 = inner[:, 0].copy()
            y0 = inner[:, 1].copy()
            a0 = np.radians(inner[:, 2])
            oa0 = oa
            side0 = best_side
        else:
            # If this run didn't improve, keep previous best params
            pass

    # --- Phase 2: refine outer angle + small position tweaks to shave side ---
    # Try a fine sweep of outer angle around best_oa
    for oa_try in np.linspace(best_oa - 10, best_oa + 10, 41):
        if time.time() > deadline - safety_margin:
            break
        side_try = _side_from_positions(best_inner, oa_try, 0.0, 0.0)
        if _check_validity(best_inner, np.array([0.0, 0.0, oa_try]), side_try):
            if side_try < best_side - 1e-9:
                best_oa = oa_try
                best_side = side_try
                best_valid = True

    # Final Nelder-Mead polish on positions only (angles fixed at 0, oa fixed)
    if time.time() < deadline - safety_margin:
        params0 = np.concatenate([best_inner[:, 0], best_inner[:, 1],
                                  np.zeros(11), [best_oa, best_side]])

        def obj_pos(params):
            x = params[0:11]
            y = params[11:22]
            inner = np.column_stack([x, y, np.zeros(11)])
            sep = _separation_violation(inner, 0.0, 0.0)
            req = _side_from_positions(inner, best_oa, 0.0, 0.0)
            pen = 0.0
            if sep > 1e-6:
                pen += 100.0 * sep + 5.0
            if best_side < req - 1e-6:
                pen += 100.0 * (req - best_side) + 5.0
            return best_side + pen

        res = minimize(obj_pos, params0, method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-7, 'fatol': 1e-7,
                                'adaptive': True, 'initial_simplex': 0.02})
        p = res.x
        inner = np.column_stack([p[0:11], p[11:22], np.zeros(11)])
        try_update(inner, best_oa)

    # --- Final verification & fallback ---
    if not _check_validity(best_inner, np.array([0.0, 0.0, best_oa]), best_side):
        # Fallback to a guaranteed valid grid
        best_inner = np.array([
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
        ], dtype=float)
        best_oa = 0.0
        best_side = 8.0

    outer_hex_data = np.array([0.0, 0.0, float(best_oa)])
    return best_inner, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END
