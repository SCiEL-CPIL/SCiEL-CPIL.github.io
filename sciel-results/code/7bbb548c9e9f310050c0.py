"""
Rigid-body cluster placement: fix 12 unit hexagons in a pre-verified dense
honeycomb cluster, then globally optimize the cluster's translation, rotation
and the outer hexagon's orientation.  This reduces the 39-D search to a
4-D continuous problem (cluster_x, cluster_y, cluster_rot, outer_rot) with
a cheap, vectorised objective.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = 145.0
SQRT3 = np.sqrt(3.0)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _hex_verts_unit(angle_rad):
    """Six vertices of a unit regular hexagon centred at origin, rotated."""
    a = angle_rad + np.radians(np.arange(0, 360, 60))
    return np.column_stack([np.cos(a), np.sin(a)])


def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return _hex_verts_unit(a) * side + np.array([cx, cy])


def _outer_normals(theta_rad):
    """Six outward face normals of a regular hexagon rotated by theta_rad."""
    a = theta_rad + np.radians(np.arange(0, 360, 60))
    return np.column_stack([np.cos(a), np.sin(a)])


def _min_side_given_verts(verts, outer_center, outer_angle_rad):
    """
    Minimum side length of a regular hexagon (centre, angle) that contains
    all points in verts.  Vectorised.
    """
    n = _outer_normals(outer_angle_rad)          # (6,2)
    rel = verts - np.array(outer_center)         # (M,2)
    proj = rel @ n.T                             # (M,6)
    # For a regular hexagon the half-width along each normal must be >= apothem
    # The hexagon is the intersection of 6 half-planes:
    #   n_k . (p - c) <= apothem   AND   -n_k . (p - c) <= apothem
    # i.e.  |n_k . (p - c)| <= apothem  for all k
    # apothem = side * sqrt(3)/2  =>  side = 2*apothem/sqrt(3)
    max_abs = np.max(np.abs(proj))               # (6,) -> max over M
    apothem = np.max(max_abs)
    return 2.0 * apothem / SQRT3


def _cluster_verts(base_positions, cluster_x, cluster_y, cluster_rot_deg):
    """
    Transform the base cluster positions (N,3: x,y,angle_deg) by
    translating by (cluster_x, cluster_y) and rotating the whole cluster
    by cluster_rot_deg about the origin.
    Returns (N*6, 2) array of all vertices.
    """
    rot = np.radians(cluster_rot_deg)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])
    pos = base_positions[:, :2].copy()
    pos = pos @ R.T + np.array([cluster_x, cluster_y])
    # angles of each hexagon also rotate
    angles = np.radians(base_positions[:, 2] + cluster_rot_deg)
    # Build all vertices
    verts_list = []
    for i in range(N):
        v = _hex_verts_unit(angles[i]) + pos[i]
        verts_list.append(v)
    return np.vstack(verts_list)


# ---------------------------------------------------------------------------
# Pre-verified dense honeycomb cluster (12 unit hexagons, no overlaps)
# ---------------------------------------------------------------------------

def _make_honeycomb_cluster():
    """
    1 centre + 6 ring-1 + 5 ring-2 (partial).
    Ring-1 spacing = 2*apothem = sqrt(3) ≈ 1.732 for edge-to-edge touching.
    We use a small gap to guarantee strict disjointness.
    """
    gap = 0.05
    d1 = SQRT3 + gap          # centre-to-ring1
    d2 = 2 * SQRT3 + gap      # centre-to-ring2 (along hex axes)
    centers = []
    centers.append([0.0, 0.0, 0.0])
    for k in range(6):
        a = np.radians(60 * k)
        centers.append([d1 * np.cos(a), d1 * np.sin(a), 0.0])
    for k in range(5):
        a = np.radians(60 * k)
        centers.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
    return np.array(centers)


def _make_circular_cluster():
    """Circular arrangement as an alternative cluster."""
    centers = []
    r = 2.2
    for i in range(N):
        a = 2 * np.pi * i / N
        centers.append([r * np.cos(a), r * np.sin(a), np.degrees(a)])
    return np.array(centers)


def _make_random_cluster(seed):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-3, 3, (N, 2))
    angles = rng.uniform(0, 60, N)
    return np.column_stack([centers, angles])


def _cluster_valid(base):
    """Check that the base cluster has no overlapping hexagons."""
    for i in range(N):
        for j in range(i + 1, N):
            vi = _hex_verts_unit(np.radians(base[i, 2])) + base[i, :2]
            vj = _hex_verts_unit(np.radians(base[j, 2])) + base[j, :2]
            if _poly_overlap(vi, vj):
                return False
    return True


def _poly_overlap(v1, v2):
    """SAT overlap test for two convex polygons (each 6 vertices)."""
    polys = [v1, v2]
    for poly in polys:
        for i in range(6):
            e = poly[(i + 1) % 6] - poly[i]
            n = np.array([-e[1], e[0]])
            nl = np.linalg.norm(n)
            if nl < 1e-12:
                continue
            n /= nl
            p1 = v1 @ n
            p2 = v2 @ n
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


# ---------------------------------------------------------------------------
# Objective for the 4-D search
# ---------------------------------------------------------------------------

def _objective(params, base_cluster, outer_center, outer_angle_rad):
    """
    params = [cx, cy, rot_deg, outer_rot_rad]
    Returns the outer side length (the quantity to minimise).
    """
    cx, cy, rot, oth = params
    verts = _cluster_verts(base_cluster, cx, cy, rot)
    return _min_side_given_verts(verts, outer_center, oth)


def _objective_full(params, base_cluster):
    """
    params = [cx, cy, rot_deg, ox, oy, outer_rot_rad]
    Jointly optimises cluster placement AND outer hexagon centre+angle.
    """
    cx, cy, rot, ox, oy, oth = params
    verts = _cluster_verts(base_cluster, cx, cy, rot)
    return _min_side_given_verts(verts, (ox, oy), oth)


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()

    # Build candidate clusters and keep only valid ones
    clusters = []
    for base in [_make_honeycomb_cluster(), _make_circular_cluster(),
                 _make_random_cluster(42), _make_random_cluster(123),
                 _make_random_cluster(999)]:
        if _cluster_valid(base):
            clusters.append(base)

    if not clusters:
        # Fallback: very spread out
        base = np.column_stack([
            np.array([3.5 * np.cos(2 * np.pi * i / N) for i in range(N)]),
            np.array([3.5 * np.sin(2 * np.pi * i / N) for i in range(N)]),
            np.zeros(N)
        ])
        clusters = [base]

    best_side = float('inf')
    best_params = None

    for base in clusters:
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 15:
            break

        # --- Stage 1: differential evolution on 6-D (cx,cy,rot,ox,oy,oth) ---
        bounds = [
            (-5, 5),    # cx
            (-5, 5),    # cy
            (0, 60),    # cluster rotation (degrees)
            (-5, 5),    # outer centre x
            (-5, 5),    # outer centre y
            (0, np.pi / 3),  # outer rotation (radians)
        ]

        try:
            res = differential_evolution(
                _objective_full, bounds,
                args=(base,),
                maxiter=200, popsize=25,
                tol=1e-6, seed=42,
                init='sobol',
                polish=True,
                time_limit=max(5, DEADLINE - (time.time() - t0) - 10),
            )
            if res.success or res.fun < best_side:
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = (base, res.x)
        except Exception:
            pass

        # --- Stage 2: local polish with Nelder-Mead / L-BFGS-B ---
        if best_params is not None:
            base2, x0 = best_params
            elapsed = time.time() - t0
            if elapsed < DEADLINE - 5:
                try:
                    res2 = minimize(
                        _objective_full, x0,
                        args=(base2,),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-10},
                    )
                    if res2.fun < best_side:
                        best_side = res2.fun
                        best_params = (base2, res2.x)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Build the final answer from best_params
    # ------------------------------------------------------------------
    if best_params is None:
        # Should not happen, but safe fallback
        base = clusters[0]
        cx = cy = 0.0
        rot = 0.0
        ox = oy = 0.0
        oth = 0.0
    else:
        base, x = best_params
        cx, cy, rot, ox, oy, oth = x

    # Compute final vertices and outer parameters
    verts = _cluster_verts(base, cx, cy, rot)
    # Recompute best outer centre & angle from the actual vertex cloud
    from scipy.optimize import minimize_scalar
    side_fn = lambda th: _min_side_given_verts(verts, (ox, oy), th)
    res_oth = minimize_scalar(side_fn, bounds=(0, np.pi / 3), method='bounded',
                              options={'xatol': 1e-8})
    oth = res_oth.x

    # Recompute outer centre for this angle
    n = _outer_normals(oth)
    proj = verts @ n.T
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    mids = (mins + maxs) / 2.0
    c_est, _, _, _ = np.linalg.lstsq(n.T, mids, rcond=None)
    ox, oy = c_est

    side = _min_side_given_verts(verts, (ox, oy), oth)

    # Build inner hexagon list in world coordinates
    rot_rad = np.radians(rot)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]])
    inner = []
    for i in range(N):
        pos = base[i, :2] @ R.T + np.array([cx, cy])
        ang = (base[i, 2] + rot) % 360
        inner.append([float(pos[0]), float(pos[1]), float(ang)])

    return inner, (float(ox), float(oy)), float(side), float(np.degrees(oth))
