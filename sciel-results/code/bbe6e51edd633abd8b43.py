# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon (6,2)."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _outer_side_and_angle(pts):
    """pts: (N,2) centered at origin. Return (min_side, best_angle_deg)."""
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 721):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        # Outward normals of the 6 edges
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = pts @ normals.T  # (N,6)
        maxproj = proj.max(axis=0)  # (6,)
        apothem = maxproj.max()
        s = 2.0 * apothem / np.sqrt(3.0)
        if s < best_s:
            best_s = s
            best_a = a_deg
    return best_s, best_a

def _compute_separations(Vs):
    """Vs: list of 11 arrays (6,2). Return (55,) array of best separation margins."""
    n_pairs = 55
    seps = np.empty(n_pairs)
    idx = 0
    for i in range(11):
        Vi = Vs[i]
        # 6 edge normals of Vi
        e_i = np.roll(Vi, -1, axis=0) - Vi
        L_i = np.hypot(e_i[:, 0], e_i[:, 1])
        n_i = np.column_stack((e_i[:, 1], -e_i[:, 0])) / L_i[:, None]  # (6,2)
        for j in range(i + 1, 11):
            Vj = Vs[j]
            e_j = np.roll(Vj, -1, axis=0) - Vj
            L_j = np.hypot(e_j[:, 0], e_j[:, 1])
            n_j = np.column_stack((e_j[:, 1], -e_j[:, 0])) / L_j[:, None]  # (6,2)
            best = -np.inf
            # Normals from i
            for k in range(6):
                n = n_i[k]
                proj_i = Vi @ n
                proj_j = Vj @ n
                lo_i, hi_i = proj_i.min(), proj_i.max()
                lo_j, hi_j = proj_j.min(), proj_j.max()
                sep = max(lo_j - hi_i, lo_i - hi_j)
                if sep > best:
                    best = sep
            # Normals from j
            for k in range(6):
                n = n_j[k]
                proj_i = Vi @ n
                proj_j = Vj @ n
                lo_i, hi_i = proj_i.min(), proj_i.max()
                lo_j, hi_j = proj_j.min(), proj_j.max()
                sep = max(lo_j - hi_i, lo_i - hi_j)
                if sep > best:
                    best = sep
            seps[idx] = best
            idx += 1
    return seps

def _objective(state):
    """state: (33,) = [x0..x10, y0..y10, a0..a10]. Returns scalar cost."""
    x = state[0:11]
    y = state[11:22]
    a = state[22:33]
    # Center
    cx = x.mean()
    cy = y.mean()
    pts = np.column_stack((x - cx, y - cy))
    s, _ = _outer_side_and_angle(pts)
    # Separation
    Vs = [_hex_vertices(x[i], y[i], 1.0, a[i]) for i in range(11)]
    seps = _compute_separations(Vs)
    # Penalty: sum of squared violations where sep <= 1e-6
    viol = np.maximum(0.0, 1e-6 - seps)
    pen = np.sum(viol ** 2)
    return s + 1e4 * pen

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 610.0  # 20s safety margin

    # Seed: hexagonal lattice arrangement (flower + ring)
    # For unit hexagons with pointy-top (angle 0), touching distance = sqrt(3)
    # Use a compact seed
    d = np.sqrt(3.0)  # center-to-center distance for touching
    seed = np.array([
        [0.0, 0.0],
        [d, 0.0],
        [-d, 0.0],
        [d/2, 1.5],
        [-d/2, 1.5],
        [d/2, -1.5],
        [-d/2, -1.5],
        [2*d, 0.0],
        [-2*d, 0.0],
        [1.5*d, 1.5],
        [-1.5*d, 1.5],
    ])
    angles0 = np.zeros(11)
    state0 = np.concatenate([seed[:, 0], seed[:, 1], angles0])

    # Bounds: positions in [-6, 6], angles in [-180, 180]
    bounds = [(-6.0, 6.0)] * 11 + [(-6.0, 6.0)] * 11 + [(-180.0, 180.0)] * 11

    best_state = state0.copy()
    best_cost = _objective(state0)

    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            polish=False,  # skip polish to save time; we'll do our own
            init='sobol',
            updating='immediate',
            callback=lambda xk, convergence: (
                (time.time() > deadline - 30.0)  # stop early if needed
            ),
        )
        if result.x is not None:
            c = _objective(result.x)
            if c < best_cost:
                best_cost = c
                best_state = result.x.copy()
    except Exception:
        pass

    # Local polish with Nelder-Mead on the best state
    if time.time() < deadline - 60.0:
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, best_state, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_cost:
                best_cost = res.fun
                best_state = res.x.copy()
        except Exception:
            pass

    # Extract final configuration
    x = best_state[0:11]
    y = best_state[11:22]
    a = best_state[22:33]
    cx = x.mean()
    cy = y.mean()
    pts = np.column_stack((x - cx, y - cy))
    s, a_outer = _outer_side_and_angle(pts)

    # Validate: check separation
    Vs = [_hex_vertices(x[i], y[i], 1.0, a[i]) for i in range(11)]
    seps = _compute_separations(Vs)
    if np.any(seps <= 1e-6):
        # Fallback: use a known valid loose packing
        d2 = 2.5
        inner = np.array([
            [0, 0, 0],
            [-d2, 0, 0],
            [d2, 0, 0],
            [-d2/2, 2.17, 0],
            [d2/2, 2.17, 0],
            [-d2/2, -2.17, 0],
            [d2/2, -2.17, 0],
            [-1.5*d2, 2.17, 0],
            [1.5*d2, 2.17, 0],
            [-1.5*d2, -2.17, 0],
            [1.5*d2, -2.17, 0],
        ], dtype=float)
        outer = np.array([0.0, 0.0, 0.0])
        return inner, outer, 8.0

    inner = np.column_stack((x, y, a))
    outer = np.array([cx, cy, a_outer])
    return inner, outer, float(s)

# EVOLVE-BLOCK-END
