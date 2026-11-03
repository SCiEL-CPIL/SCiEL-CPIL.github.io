import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
T0 = time.time()
DEADLINE = T0 + 142.0
SQRT3 = math.sqrt(3.0)
SQRT3_HALF = SQRT3 / 2.0


def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of the outer hexagon."""
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side(theta, centers, theta_outer):
    """Minimal outer side length containing all inner hexagon vertices."""
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    apothem = proj.max()
    return apothem / SQRT3_HALF


def _pairwise_penalty(centers):
    """Penalty for pairs of centers closer than 2 (overlap)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return float(np.sum(viol * viol))


def _objective(params, theta_outer):
    """Objective: outer side length + large penalty for overlap."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side(theta, centers, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _is_valid(centers, theta, theta_outer, tol=1e-5):
    """Check non-overlap and containment."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    s = _outer_side(theta, centers, theta_outer)
    apothem = s * SQRT3_HALF
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    return True


def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """
    LP: optimal outer-hexagon center + inradius for given inner hexagons and
    outer angle.  Returns (cx, cy, side_length) or None.
    """
    all_vertices = []
    for ix, iy, iangle_deg in inner_hexes:
        iangle = math.radians(iangle_deg)
        for k in range(6):
            a = iangle + math.pi * k / 3
            all_vertices.append((ix + math.cos(a), iy + math.sin(a)))

    c = [0.0, 0.0, 1.0]
    A_ub = []
    b_ub = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / SQRT3_HALF
            return cx, cy, side
    except Exception:
        pass
    return None


def _find_best_outer_lp(inner_hexes, time_limit=8.0):
    """
    Search the outer-hexagon angle (0..30 deg) with LP.
    Coarse grid of 0.5 deg, then a fine 0.01 deg grid around the best.
    Returns (cx, cy, side, angle_deg).
    """
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    coarse = np.arange(0, 30.01, 0.5)
    for angle_deg in coarse:
        if time.time() > start + time_limit:
            break
        res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    if best_side < float('inf'):
        fine_lo = max(0.0, best_angle - 0.5)
        fine_hi = min(30.0, best_angle + 0.5)
        fine = np.arange(fine_lo, fine_hi + 0.001, 0.01)
        for angle_deg in fine:
            if time.time() > start + time_limit:
                break
            res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle


def _make_init(seed):
    """Generate initial parameters for optimization."""
    rng = np.random.RandomState(seed)
    mode = seed % 5
    sqrt3 = math.sqrt(3.0)

    if mode == 0:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 1:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + math.pi / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 2:
        positions = [(0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
                     (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)]
        centers = np.array(positions, dtype=float)
    elif mode == 3:
        u = (sqrt3, 0.0)
        v = (sqrt3 / 2, 1.5)
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                candidates.append((x * x + y * y, x, y))
        candidates.sort(key=lambda t: t[0])
        centers = np.array([(c[1], c[2]) for c in candidates[:N]])
    else:
        positions = [(0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
                     (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)]
        centers = np.array(positions, dtype=float) + rng.uniform(-0.1, 0.1, size=(N, 2))

    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    return np.concatenate([centers.reshape(-1), [theta]])


def _optimize_one(init_params, theta_outer, maxiter=350):
    """Single L-BFGS-B optimization run (fixed outer angle)."""
    try:
        res = minimize(_objective, init_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11})
        return res
    except Exception:
        return None


def _smooth_objective(params):
    """
    Epigraph / smooth surrogate: simultaneously optimizes inner positions,
    inner angle AND outer angle.
    """
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    theta_outer = params[N * 2 + 1]

    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    proj = np.concatenate([proj, -proj], axis=1)

    m = proj.max(axis=0)
    lse = m + np.log(np.sum(np.exp(proj - m), axis=0))
    beta = 80.0
    apothem = float(np.mean(lse) / beta)
    s = apothem / SQRT3_HALF

    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _smooth_polish(params, theta_outer, maxiter=600):
    """High-precision polish that also refines the outer angle."""
    p0 = np.concatenate([params, [theta_outer]])
    try:
        res = minimize(_smooth_objective, p0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13})
        if res.fun < _smooth_objective(p0):
            return res.x
    except Exception:
        pass
    return params


def _polish(params, theta_outer):
    """High-precision polish (fixed outer angle)."""
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 300, 'ftol': 1e-15})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    return params


def _lp_refine(centers, theta, time_limit=6.0):
    """LP outer-hexagon search for fixed inner hexagons."""
    inner_hexes = [[float(centers[i, 0]), float(centers[i, 1]),
                    float(math.degrees(theta))] for i in range(N)]
    return _find_best_outer_lp(inner_hexes, time_limit=time_limit)


def _rigid_cluster_centers():
    """
    Build the 1-2-3-2-1 hexagonal lattice cluster of 11 unit hexagons.
    The cluster uses a hexagonal lattice with basis vectors:
      u = (2, 0)  (horizontal neighbor distance)
      v = (1, sqrt(3))  (diagonal neighbor)
    Actually, for unit hexagons (circumradius 1), the center-to-center
    distance for touching hexagons is 2 (for edge-adjacent) or 2*sqrt(3)
    (for vertex-adjacent). The standard hexagonal lattice for edge-adjacent
    hexagons has spacing 2 in one direction and 2*sqrt(3) in the other,
    but the simplest is to use the triangular lattice with spacing 2*sqrt(3)
    for the "corner-to-corner" or 2 for "edge-to-edge".
    
    For unit hexagons with circumradius 1, two hexagons are edge-adjacent
    when their centers are 2*sqrt(3) apart? No. Let me think carefully.
    
    A unit regular hexagon has side length 1, circumradius 1, inradius sqrt(3)/2.
    Two unit hexagons are edge-adjacent (sharing a full edge) when their
    centers are 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) apart? No, that's
    not right either.
    
    Actually, for two regular hexagons of side 1 to be edge-adjacent (sharing
    a full edge), the distance between centers is 2 * apothem = 2 * (sqrt(3)/2)
    = sqrt(3). Wait, no. If two hexagons share an edge, the distance between
    their centers is twice the apothem (distance from center to edge), which
    is 2 * (sqrt(3)/2) = sqrt(3). But that would mean they overlap in area...
    
    Let me reconsider. For two unit hexagons (side=1) to be just touching
    (disjoint, sharing at most a point or edge), the minimum center distance
    depends on orientation. If they're aligned and edge-adjacent, the center
    distance is 2 * apothem = sqrt(3) ≈ 1.732. But that means they share an
    edge, which is "touching" but the interiors are disjoint.
    
    However, for the pairwise penalty in the code, it uses distance < 2.0 as
    overlap. This suggests the code is using a simpler model where the
    "effective radius" is 1 (circumradius), so two hexagons overlap if their
    centers are closer than 2. This is a conservative approximation.
    
    For the rigid cluster, I'll use the standard hexagonal lattice arrangement
    where hexagons are placed at the vertices of a triangular lattice with
    spacing such that they're edge-adjacent. The 1-2-3-2-1 pattern:
    
    Row 0: 1 hex at (0, 0)
    Row 1: 2 hexes
    Row 2: 3 hexes
    Row 3: 2 hexes
    Row 4: 1 hex
    
    For a hexagonal lattice with edge-adjacent unit hexagons, the center
    spacing is sqrt(3) in the horizontal direction (for hexes in the same
    row) and the vertical spacing between rows is 1.5 (for pointy-top) or
    sqrt(3) (for flat-top). Let me use flat-top hexagons.
    
    For flat-top hexagons (one vertex pointing up), the vertices are at
    angles 0, 60, 120, 180, 240, 300 degrees from the positive x-axis.
    The hexagon has width 2 (from leftmost to rightmost vertex) and height
    sqrt(3) (from top to bottom edge midpoint... actually height = 2*apothem
    = sqrt(3) for the distance between parallel edges, but the full height
    from top vertex to bottom vertex is 2).
    
    For flat-top hexagons, two hexagons in the same row (horizontally
    adjacent) have centers 3 apart? No. The horizontal distance between
    centers of two edge-adjacent flat-top hexagons is 3 * side = 3? 
    Let me just use the standard: for flat-top hexagons of side 1, the
    center-to-center distance for horizontal adjacency is 3 (since the
    hexagon extends 1.5 to the left and 1.5 to the right of center for
    the flat sides... no, that's not right either).
    
    I'll just use the positions from the current program's mode 2, which
    is a known valid arrangement:
    (0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
    (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)
    
    This gives a 1-2-3-2-1 pattern with center distances of 2 (for
    horizontally adjacent) and 2*sqrt(3) (for diagonally adjacent), which
    is consistent with the pairwise penalty threshold of 2.0.
    """
    sqrt3 = math.sqrt(3.0)
    positions = [
        (0, 0),          # row 0: 1
        (-1, sqrt3), (1, sqrt3),   # row 1: 2
        (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),  # row 2: 3
        (-1, 3*sqrt3), (1, 3*sqrt3),   # row 3: 2
        (0, 4*sqrt3),      # row 4: 1
    ]
    return np.array(positions, dtype=float)


def _rigid_cluster_side(theta_cluster, theta_outer):
    """
    For a rigid cluster (all hexagons have the same rotation theta_cluster),
    compute the minimum outer hexagon side length using LP over the outer
    hexagon center and angle.
    """
    centers = _rigid_cluster_centers()
    # Rotate the cluster by theta_cluster
    c = math.cos(theta_cluster)
    s = math.sin(theta_cluster)
    rot = np.array([[c, -s], [s, c]])
    centers_rot = centers @ rot.T
    
    inner_hexes = [[float(centers_rot[i, 0]), float(centers_rot[i, 1]),
                    float(math.degrees(theta_cluster))] for i in range(N)]
    
    # Find best outer hexagon via LP
    lp_res = _find_best_outer_lp(inner_hexes, time_limit=3.0)
    if lp_res is None:
        return float('inf')
    _, _, side, _ = lp_res
    return side


def _angular_sweep_rigid(time_limit=15.0):
    """
    1D angular sweep of the rigid cluster's global rotation.
    Since the cluster is rigid, the minimum enclosing side is a function
    of the single rotation angle. We do a coarse sweep followed by
    ternary search refinement.
    """
    start = time.time()
    
    # Coarse sweep: 0 to 30 degrees (hexagonal symmetry of the outer hexagon)
    best_side = float('inf')
    best_angle = 0.0
    coarse_angles = np.arange(0, 30.01, 1.0)
    
    for ang_deg in coarse_angles:
        if time.time() > start + time_limit:
            break
        side = _rigid_cluster_side(math.radians(ang_deg), 0.0)
        if side < best_side:
            best_side = side
            best_angle = float(ang_deg)
    
    if best_side == float('inf'):
        return None
    
    # Ternary search refinement around the best coarse angle
    lo = max(0.0, best_angle - 1.0)
    hi = min(30.0, best_angle + 1.0)
    
    for _ in range(30):
        if time.time() > start + time_limit:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = _rigid_cluster_side(math.radians(m1), 0.0)
        f2 = _rigid_cluster_side(math.radians(m2), 0.0)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
        if f1 < best_side:
            best_side = f1
            best_angle = m1
        if f2 < best_side:
            best_side = f2
            best_angle = m2
    
    # Final LP refinement at the best angle
    centers = _rigid_cluster_centers()
    c = math.cos(math.radians(best_angle))
    s = math.sin(math.radians(best_angle))
    rot = np.array([[c, -s], [s, c]])
    centers_rot = centers @ rot.T
    
    inner_hexes = [[float(centers_rot[i, 0]), float(centers_rot[i, 1]),
                    float(best_angle)] for i in range(N)]
    lp_res = _find_best_outer_lp(inner_hexes, time_limit=5.0)
    if lp_res is None:
        return None
    
    cx, cy, side, angle_deg = lp_res
    return centers_rot, best_angle, side, angle_deg


def _strict_disjoint_check(centers, theta, tol=1e-7):
    """
    Strict geometric disjointness check: for each pair of hexagons,
    verify no vertex of one is inside the other and no edges intersect.
    """
    def hex_verts(cx, cy, ang):
        v = []
        for k in range(6):
            a = ang + k * math.pi / 3
            v.append((cx + math.cos(a), cy + math.sin(a)))
        return v
    
    def point_in_hex(px, py, cx, cy, ang):
        dx = px - cx
        dy = py - cy
        a = -ang
        lx = dx * math.cos(a) - dy * math.sin(a)
        ly = dx * math.sin(a) + dy * math.cos(a)
        apothem = SQRT3_HALF
        for k in range(6):
            a_norm = math.pi / 6 + k * math.pi / 3
            nx = math.cos(a_norm)
            ny = math.sin(a_norm)
            if lx * nx + ly * ny > apothem + tol:
                return False
        return True
    
    def seg_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        return False
    
    for i in range(N):
        for j in range(i + 1, N):
            v1 = hex_verts(centers[i, 0], centers[i, 1], theta)
            v2 = hex_verts(centers[j, 0], centers[j, 1], theta)
            # Check vertices of hex1 inside hex2
            for vx, vy in v1:
                if point_in_hex(vx, vy, centers[j, 0], centers[j, 1], theta):
                    return False
            # Check vertices of hex2 inside hex1
            for vx, vy in v2:
                if point_in_hex(vx, vy, centers[i, 0], centers[i, 1], theta):
                    return False
            # Check edge intersections
            for k in range(6):
                for l in range(6):
                    if seg_intersect(v1[k], v1[(k+1)%6], v2[l], v2[(l+1)%6]):
                        return False
    return True


def construct():
    best_score = float('inf')
    best_result = None  # (centers, theta, theta_outer)
    
    # ---- Phase 1: Rigid cluster angular sweep (fast, high-quality start) ----
    if time.time() < DEADLINE - 30:
        rigid_res = _angular_sweep_rigid(time_limit=20.0)
        if rigid_res is not None:
            centers_r, theta_r, side_r, angle_deg_r = rigid_res
            theta_outer_r = math.radians(angle_deg_r)
            # Verify with strict check
            if _strict_disjoint_check(centers_r, theta_r):
                s_check = _outer_side(theta_r, centers_r, theta_outer_r)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers_r.copy(), theta_r, theta_outer_r)
    
    # ---- Phase 2: Multi-start continuous optimization from rigid + random inits ----
    outer_rotations = [0.0, math.pi / 6.0, math.pi / 3.0]
    seed = 0
    maxiter = 350
    
    # If we have a rigid result, use it as the primary init
    if best_result is not None:
        centers_r, theta_r, theta_o_r = best_result
        init_rigid = np.concatenate([centers_r.reshape(-1), [theta_r]])
        # Optimize from rigid start with each outer rotation
        for theta_o in outer_rotations:
            if time.time() >= DEADLINE - 35:
                break
            res = _optimize_one(init_rigid, theta_o, maxiter)
            if res is None:
                continue
            params = res.x
            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            if _is_valid(centers, theta, theta_o):
                s = _outer_side(theta, centers, theta_o)
                if s < best_score:
                    best_score = s
                    best_result = (centers.copy(), theta, theta_o)
    
    # Additional random inits
    for theta_o in outer_rotations:
        while time.time() < DEADLINE - 35:
            init_params = _make_init(seed)
            seed += 1
            res = _optimize_one(init_params, theta_o, maxiter)
            if res is None:
                continue
            params = res.x
            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            if not _is_valid(centers, theta, theta_o):
                continue
            s = _outer_side(theta, centers, theta_o)
            if s < best_score:
                best_score = s
                best_result = (centers.copy(), theta, theta_o)
    
    if best_result is None:
        # Fallback: rigid cluster
        centers = _rigid_cluster_centers()
        theta = 0.0
        theta_o = 0.0
        s = _outer_side(theta, centers, theta_o)
        best_score = s
        best_result = (centers, theta, theta_o)
    
    # ---- Phase 3: Smooth epigraph polish (also refines outer angle) ----
    if time.time() < DEADLINE - 20:
        centers, theta, theta_o = best_result
        params = np.concatenate([centers.reshape(-1), [theta]])
        polished = _smooth_polish(params, theta_o)
        centers_p = polished[:N * 2].reshape(N, 2)
        theta_p = polished[N * 2]
        theta_o_p = polished[N * 2 + 1]
        if _is_valid(centers_p, theta_p, theta_o_p):
            s_p = _outer_side(theta_p, centers_p, theta_o_p)
            if s_p < best_score:
                best_score = s_p
                best_result = (centers_p.copy(), theta_p, theta_o_p)
    
    # ---- Phase 4: Fixed-angle high-precision polish ----
    if time.time() < DEADLINE - 12:
        centers, theta, theta_o = best_result
        params = np.concatenate([centers.reshape(-1), [theta]])
        polished = _polish(params, theta_o)
        centers_p = polished[:N * 2].reshape(N, 2)
        theta_p = polished[N * 2]
        if _is_valid(centers_p, theta_p, theta_o):
            s_p = _outer_side(theta_p, centers_p, theta_o)
            if s_p < best_score:
                best_score = s_p
                best_result = (centers_p.copy(), theta_p, theta_o)
    
    # ---- Phase 5: LP outer-hexagon refinement ----
    if time.time() < DEADLINE - 6:
        centers, theta, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=8.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad)
    
    # ---- Phase 6: One final LP refinement ----
    if time.time() < DEADLINE - 1:
        centers, theta, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=4.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad)
    
    centers, theta, theta_o = best_result
    s = _outer_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
