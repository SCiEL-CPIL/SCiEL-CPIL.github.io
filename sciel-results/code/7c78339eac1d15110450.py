"""
Packing 12 unit hexagons inside a regular hexagon.
Approach:
  1. Build a hexagonal-lattice initial configuration (19-hex cluster, drop 7).
  2. Compute the Minkowski-sum forbidden zone for two unit hexagons at any relative angle.
  3. Use a smooth objective: max over (i,j) of (distance_required(angle_ij) - actual_dist)^2
     plus an outer-side-length term, optimised with Nelder-Mead from multiple starts.
  4. Return the best valid (no-overlap) configuration.

Interface: construct() -> (inner, center, side, angle)
  inner: list of 12 rows [x, y, angle_degrees]
  center: (x, y) of outer hexagon
  side: outer side length
  angle: outer rotation (radians)
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)
APOTHEM = SQRT3 / 2.0

# Unit hexagon vertices (pointy-top by default, i.e. first vertex at 90°)
# We use flat-top: first vertex at 0°
UNIT_VERTS = np.array([
    [math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))]
    for i in range(6)
], dtype=np.float64)

# Edge outward normals for flat-top hexagon (edges at 30, 90, 150, 210, 270, 330 deg)
EDGE_NORMALS = np.array([
    [math.cos(math.radians(30 + 60 * i)), math.sin(math.radians(30 + 60 * i))]
    for i in range(6)
], dtype=np.float64)


def _hex_vertices(cx, cy, angle_rad, side=1.0):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return np.array([cx, cy]) + side * (UNIT_VERTS @ R.T)


def _hex_normals(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return EDGE_NORMALS @ R.T


def _minkowski_vertices(angle_diff_rad):
    """
    Vertices of the Minkowski sum of two unit hexagons, one at angle 0 and one at angle_diff.
    The Minkowski sum of two convex polygons with m and n vertices has at most m+n vertices.
    We compute it by merging the edge vectors sorted by angle.
    """
    # Edge vectors of hexagon at angle 0
    v1 = np.roll(UNIT_VERTS, -1, axis=0) - UNIT_VERTS  # edge vectors
    # Edge vectors of hexagon at angle_diff
    c, s = math.cos(angle_diff_rad), math.sin(angle_diff_rad)
    R = np.array([[c, -s], [s, c]])
    v2 = (R @ UNIT_VERTS.T).T
    v2 = np.roll(v2, -1, axis=0) - v2

    # Convert to polar (angle, length) and merge
    edges = []
    for e in v1:
        angle = math.atan2(e[1], e[0])
        edges.append((angle, e[0], e[1]))
    for e in v2:
        angle = math.atan2(e[1], e[0])
        edges.append((angle, e[0], e[1]))
    edges.sort(key=lambda x: x[0])

    # Build Minkowski sum polygon starting from the sum of the "lowest-left" vertices
    # The Minkowski sum of two polygons A+B: start at min_A + min_B (lexicographic),
    # then walk edges in angle order.
    # For two centered hexagons, the Minkowski sum is also centered at origin if both are centered.
    # Actually the Minkowski sum of two centrally-symmetric convex bodies is centrally symmetric.
    # So the sum is centered at (0,0) if both are centered at origin.

    # Start point: the vertex of A with smallest y (then x) plus vertex of B with smallest y (then x)
    # For a regular hexagon centered at origin, the lowest vertex is at angle 270° (or -90°) for pointy-top,
    # but for flat-top the lowest points are edges. Let's just compute by walking.

    # Simpler: the Minkowski sum of two regular hexagons (both centered at origin) is a regular 12-gon
    # only when angle_diff is a multiple of 30°. In general it's a convex 12-gon (or fewer if some edges align).
    # We can compute it by taking all 12 vertices (6 from each, offset) and convex-hull'ing.

    # Vertices of hex1 (angle 0): UNIT_VERTS
    # Vertices of hex2 (angle_diff): R @ UNIT_VERTS.T
    all_verts = np.vstack([UNIT_VERTS, (R @ UNIT_VERTS.T).T])
    # Convex hull via simple gift-wrapping or use scipy
    # For 12 points, a quick convex hull:
    pts = all_verts
    # Find leftmost point
    idx = np.argmin(pts[:, 0] * 1e9 + pts[:, 1])
    hull = [pts[idx]]
    cur = idx
    for _ in range(len(pts)):
        nxt = (cur + 1) % len(pts)
        best = nxt
        best_cross = -np.inf
        for cand in range(len(pts)):
            if cand == cur:
                continue
            cross = (pts[cand, 0] - pts[cur, 0]) * (pts[best, 1] - pts[cur, 1]) - \
                    (pts[cand, 1] - pts[cur, 1]) * (pts[best, 0] - pts[cur, 0])
            # We want the most counter-clockwise turn
            angle1 = math.atan2(pts[cand, 1] - pts[cur, 1], pts[cand, 0] - pts[cur, 0])
            angle2 = math.atan2(pts[best, 1] - pts[cur, 1], pts[best, 0] - pts[cur, 0])
            # Normalize to [0, 2pi)
            a1 = angle1 % (2 * math.pi)
            a2 = angle2 % (2 * math.pi)
            if a1 < a2:
                best = cand
        if best == cur or len(hull) > len(pts):
            break
        hull.append(pts[best])
        cur = best

    hull = np.array(hull)
    # The Minkowski sum of two centered hexagons is centered at origin.
    # But our construction may not be perfectly centered; shift to center.
    center = hull.mean(axis=0)
    hull = hull - center
    return hull


# Precompute Minkowski sums for a grid of angle differences
_N_ANGLES = 720  # 0.5 degree resolution
_ANGLE_GRID = np.linspace(0, 2 * math.pi, _N_ANGLES, endpoint=False)
_MINK_SUMS = [None] * _N_ANGLES

# We'll compute support function h(theta) of the Minkowski sum for each angle difference.
# The distance required for non-overlap at relative angle d is the support function of the
# Minkowski sum in the direction of the center-to-center vector.

# Precompute for each angle_diff in the grid, the support function h(theta) sampled at 720 thetas.
_THETA_GRID = _ANGLE_GRID  # same grid
_SUPPORT = np.zeros((_N_ANGLES, _N_ANGLES))

def _compute_support():
    for i, ad in enumerate(_ANGLE_GRID):
        # Get Minkowski sum vertices for this angle diff
        c, s = math.cos(ad), math.sin(ad)
        R = np.array([[c, -s], [s, c]])
        v2 = (R @ UNIT_VERTS.T).T
        all_v = np.vstack([UNIT_VERTS, v2])
        # Convex hull of 12 points
        # Use scipy if available, otherwise manual
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(all_v)
            hv = all_v[hull.vertices]
        except Exception:
            # Fallback: all 12 points (not all are on hull, but overestimates support slightly)
            hv = all_v
        # Support function: h(theta) = max dot(v, [cos(theta), sin(theta)])
        dirs = np.column_stack([np.cos(_THETA_GRID), np.sin(_THETA_GRID)])
        _SUPPORT[i] = hv @ dirs.T

_compute_support()


def _required_distance(angle_diff_rad):
    """
    Return the minimum center-to-center distance for two unit hexagons
    with relative angle angle_diff_rad to not overlap.
    This is the support function of the Minkowski sum in the direction of the
    center-to-center vector. We don't know the direction a priori, so for the
    optimization we use the worst-case over all directions? No — the direction
    is determined by the actual center difference.
    Instead, in the objective we'll compute the actual penetration.
    """
    pass  # Not used directly; see _overlap_penalty


def _sat_penetration(v1, n1, v2, n2):
    """
    Compute the minimum translation distance to separate two convex hexagons.
    Returns 0 if not overlapping, or the penetration depth if overlapping.
    Uses SAT: for each edge normal, compute the overlap; the minimum overlap
    across separating axes is the penetration depth (if all overlaps > 0).
    """
    min_overlap = float('inf')
    for k in range(6):
        for axis in (n1[k], n2[k]):
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
            if overlap < 0:
                return 0.0  # Separating axis found
            if overlap < min_overlap:
                min_overlap = overlap
    return min_overlap


def _is_valid(centers, angles_rad, eps=1e-7):
    """Check no pair overlaps using SAT."""
    n = len(centers)
    all_v = []
    all_n = []
    for i in range(n):
        all_v.append(_hex_vertices(centers[i][0], centers[i][1], angles_rad[i]))
        all_n.append(_hex_normals(angles_rad[i]))
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if dx * dx + dy * dy > 4.5:
                continue
            pen = _sat_penetration(all_v[i], all_n[i], all_v[j], all_n[j])
            if pen > eps:
                return False
    return True


def _calc_L(centers, angles_rad):
    """Minimum outer hexagon side length (unrotated, center at origin)."""
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles_rad[i])
        projs = verts @ EDGE_NORMALS.T
        local_max = projs.max()
        if local_max > max_proj:
            max_proj = local_max
    return 2.0 * max_proj / SQRT3


def _smooth_objective(params, outer_side=None):
    """
    Smooth objective: sum of squared penetration depths + (outer_side_penalty).
    params: [x0,y0,ang0, x1,y1,ang1, ..., x11,y11,ang11]
    Returns a scalar; lower is better. Overlap penalty dominates.
    """
    centers = [(params[2 * i], params[2 * i + 1]) for i in range(N)]
    angles = [params[2 * N + i] for i in range(N)]

    all_v = []
    all_n = []
    for i in range(N):
        all_v.append(_hex_vertices(centers[i][0], centers[i][1], angles[i]))
        all_n.append(_hex_normals(angles[i]))

    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d2 = dx * dx + dy * dy
            if d2 > 4.5:
                continue
            pen = _sat_penetration(all_v[i], all_n[i], all_v[j], all_n[j])
            if pen > 0:
                penalty += pen * pen
    if penalty > 1e-4:
        return penalty * 1000.0 + _calc_L(centers, angles)

    L = _calc_L(centers, angles)
    return L


def _build_hex_cluster_init():
    """
    Build a 19-hexagonal-cluster initial configuration, then remove 7 hexagons
    to get 12. We remove the 7 outermost in a pattern that keeps the cluster compact.
    Lattice vectors for flat-top hex grid:
      a1 = (sqrt(3), 0)
      a2 = (sqrt(3)/2, 3/2)
    19-hex cluster: all (q,r) with |q|<=2, |r|<=2, |q+r|<=2
    """
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])

    cluster = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q + r) <= 2:
                pt = q * a1 + r * a2
                cluster.append((pt[0], pt[1], q, r))

    # cluster has 19 entries. Remove 7 to get 12.
    # Remove the 7 with the largest distance from center, breaking ties by (q,r).
    cluster.sort(key=lambda x: (-(x[0] ** 2 + x[1] ** 2), x[2], x[3]))
    # Take the 12 closest
    selected = cluster[:12]
    centers = [(c[0], c[1]) for c in selected]
    angles = [0.0] * 12
    return centers, angles


def _build_ring_init():
    """
    Alternative init: 1 central hex + 11 around it in two rings.
    Ring 1: 6 hexagons at distance sqrt(3) around center.
    Ring 2: 5 hexagons placed in gaps of ring 1.
    """
    centers = [(0.0, 0.0)]
    angles = [0.0]
    d = SQRT3  # center-to-center for side-to-side contact
    for k in range(6):
        theta = math.pi / 3 * k
        centers.append((d * math.cos(theta), d * math.sin(theta)))
        angles.append(0.0)
    # 5 more: place at distance ~2*sqrt(3) in alternating directions
    d2 = 2 * SQRT3
    for k in range(5):
        theta = math.pi / 3 * (2 * k + 1)  # odd multiples of 60 deg
        centers.append((d2 * math.cos(theta), d2 * math.sin(theta)))
        angles.append(0.0)
    return centers, angles


def _random_init(scale=3.5):
    """Random positions within a hexagon of side ~scale, random angles."""
    centers = []
    angles = []
    for i in range(N):
        # Random point in a hexagon of side R centered at origin
        R = scale
        while True:
            x = (np.random.rand() - 0.5) * 2 * R
            y = (np.random.rand() - 0.5) * 2 * R
            # Check if inside hexagon: |x| <= R, |y| <= R*sqrt(3)/2,
            # and |x| + |y|/sqrt(3) <= R (for flat-top hex)
            if abs(x) <= R and abs(y) <= R * SQRT3 / 2 and abs(x) + abs(y) / SQRT3 <= R:
                break
        centers.append((x, y))
        angles.append(np.random.rand() * 2 * math.pi)
    return centers, angles


def _optimize_from(centers, angles, max_iter=2000, xatol=1e-5, fatol=1e-5, deadline=None):
    """Run Nelder-Mead from the given init. Return (centers, angles, L) if valid."""
    init = []
    for i in range(N):
        init.extend([centers[i][0], centers[i][1], angles[i]])

    def obj(p):
        if deadline is not None and time.time() > deadline:
            return 1e6
        return _smooth_objective(p)

    try:
        res = minimize(
            obj, init, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
    except Exception:
        return None

    new_centers = [(res.x[2 * i], res.x[2 * i + 1]) for i in range(N)]
    new_angles = [res.x[2 * N + i] for i in range(N)]

    if _is_valid(new_centers, new_angles, eps=1e-6):
        L = _calc_L(new_centers, new_angles)
        return new_centers, new_angles, L
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    best_L = float('inf')
    best_centers = None
    best_angles = None

    def try_config(centers, angles):
        nonlocal best_L, best_centers, best_angles
        if _is_valid(centers, angles, eps=1e-6):
            L = _calc_L(centers, angles)
            if L < best_L:
                best_L = L
                best_centers = [c[:] for c in centers]
                best_angles = angles[:]

    # Phase 1: Evaluate several analytical initial configurations
    inits = []
    c1, a1 = _build_hex_cluster_init()
    inits.append((c1, a1))
    c2, a2 = _build_ring_init()
    inits.append((c2, a2))

    # A few random inits
    for _ in range(4):
        cr, ar = _random_init(scale=3.0)
        inits.append((cr, ar))

    for centers, angles in inits:
        try_config(centers, angles)

    # Phase 2: Continuous optimization from each initial configuration
    for centers, angles in inits:
        if time.time() > deadline - 30:
            break
        result = _optimize_from(centers, angles, max_iter=3000, xatol=1e-5, fatol=1e-5,
                                deadline=deadline - 15)
        if result is not None:
            nc, na, L = result
            if L < best_L:
                best_L = L
                best_centers = nc
                best_angles = na

    # Phase 3: Additional random restarts if time permits
    restart_count = 0
    while time.time() < deadline - 20 and restart_count < 8:
        restart_count += 1
        cr, ar = _random_init(scale=2.8 + np.random.rand() * 0.5)
        try_config(cr, ar)
        result = _optimize_from(cr, ar, max_iter=1500, xatol=1e-5, fatol=1e-5,
                                deadline=deadline - 10)
        if result is not None:
            nc, na, L = result
            if L < best_L:
                best_L = L
                best_centers = nc
                best_angles = na

    # Phase 4: Local perturbation around best solution
    if best_centers is not None and time.time() < deadline - 10:
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            pc = []
            pa = []
            for i in range(N):
                dx = (np.random.rand() - 0.5) * 0.15
                dy = (np.random.rand() - 0.5) * 0.15
                da = (np.random.rand() - 0.5) * 0.1
                pc.append((best_centers[i][0] + dx, best_centers[i][1] + dy))
                pa.append(best_angles[i] + da)
            result = _optimize_from(pc, pa, max_iter=1000, xatol=1e-5, fatol=1e-5,
                                    deadline=deadline - 5)
            if result is not None:
                nc, na, L = result
                if L < best_L:
                    best_L = L
                    best_centers = nc
                    best_angles = na

    # Fallback
    if best_centers is None:
        centers, angles = _build_hex_cluster_init()
        if _is_valid(centers, angles, eps=1e-6):
            best_centers = centers
            best_angles = angles
            best_L = _calc_L(centers, angles)
        else:
            # Spread on a large circle
            best_centers = []
            best_angles = [0.0] * N
            for i in range(N):
                theta = 2 * math.pi * i / N
                r = 5.0
                best_centers.append((r * math.cos(theta), r * math.sin(theta)))
            best_L = _calc_L(best_centers, best_angles)

    # Final check
    if not _is_valid(best_centers, best_angles, eps=1e-6):
        # Try the hex cluster again with a looser eps
        centers, angles = _build_hex_cluster_init()
        if _is_valid(centers, angles, eps=1e-5):
            best_centers = centers
            best_angles = angles
            best_L = _calc_L(centers, angles)
        else:
            best_centers = []
            best_angles = [0.0] * N
            for i in range(N):
                theta = 2 * math.pi * i / N
                r = 6.0
                best_centers.append((r * math.cos(theta), r * math.sin(theta)))
            best_L = _calc_L(best_centers, best_angles)

    final_L = _calc_L(best_centers, best_angles)
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])

    return inner, (0.0, 0.0), final_L, 0.0
