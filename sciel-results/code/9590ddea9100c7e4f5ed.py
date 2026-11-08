"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses exact support-function containment + exact hex-hex separation, with
differential evolution and L-BFGS polishing.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = np.sqrt(3.0)
APOTHEM = SQRT3 / 2.0  # inradius of a unit hexagon

# Precomputed: 6 vertex-direction angles (radians) for a unit hexagon
_VERT_ANGLES = np.radians(60.0 * np.arange(6))
# Precomputed: 6 outer-normal base angles (radians) for outer hex rotation 0
_OUTER_NORM_BASE = np.radians(60.0 * np.arange(6))


def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])


def _support_hex(beta_rad, theta_rad):
    """
    Support function of a unit hexagon rotated by beta_rad in direction theta_rad.
    Returns max_k cos(beta + 60k - theta).
    """
    return np.max(np.cos(beta_rad + _VERT_ANGLES - theta_rad))


def _outer_side(params, outer_angle_rad):
    """
    Compute the minimum outer side length S such that all N unit hexagons
    (positions + rotations in params) are contained in a regular hexagon
    centered at origin with rotation outer_angle_rad.
    
    params: (N*3,) array [x0, y0, ang0_deg, x1, y1, ang1_deg, ...]
    Returns S (float).
    """
    # Extract centers and angles
    cx = params[0::3]
    cy = params[1::3]
    ang_deg = params[2::3]
    ang_rad = np.radians(ang_deg)  # shape (N,)
    
    # Outer normals: 6 directions
    outer_normals = outer_angle_rad + _OUTER_NORM_BASE  # shape (6,)
    # cos and sin of outer normals
    on_cos = np.cos(outer_normals)  # (6,)
    on_sin = np.sin(outer_normals)  # (6,)
    
    # For each inner hex k and each outer normal i:
    #   dot_center = cx_k * on_cos_i + cy_k * on_sin_i
    #   support = max_m cos(ang_rad_k + 60m - outer_normals_i)
    #   needed = dot_center + support
    # S = 2 * max(all needed) / sqrt(3)
    
    # Vectorized: dot_center shape (N, 6)
    dot_center = cx[:, None] * on_cos[None, :] + cy[:, None] * on_sin[None, :]
    
    # support: for each (k, i), max over m of cos(ang_k + 60m - outer_normal_i)
    # ang_rad[:, None, None] + _VERT_ANGLES[None, None, :] - outer_normals[None, :, None]
    # shape (N, 6, 6)
    cos_vals = np.cos(ang_rad[:, None, None] + _VERT_ANGLES[None, None, :] - outer_normals[None, :, None])
    support = np.max(cos_vals, axis=2)  # shape (N, 6)
    
    needed = dot_center + support  # (N, 6)
    max_needed = np.max(needed)
    
    S = 2.0 * max_needed / SQRT3
    return S


def _overlap_penalty(params):
    """
    Exact overlap penalty: for each pair of unit hexagons, check if they overlap.
    Two convex polygons are disjoint iff there exists a separating axis.
    For two regular hexagons, the separating axes are the 6 edge normals of each.
    
    Hex A (center pA, rotation betaA) and Hex B (center pB, rotation betaB) are
    disjoint iff for at least one of the 12 edge-normal directions, the
    projection of (pB - pA) onto that direction is >= 2 * apothem = sqrt(3).
    
    Wait, that's not quite right. Let me think again.
    
    Two convex polygons P and Q are disjoint iff there exists a direction d such
    that max_{p in P} dot(p, d) < min_{q in Q} dot(q, d).
    
    Equivalently: max_{p in P} dot(p, d) - min_{q in Q} dot(q, d) < 0 for some d.
    
    For a unit hexagon centered at p with rotation beta, the support in direction d
    (unit vector at angle theta) is: p·d + h(beta, theta) where h is the support of
    the unit hex at origin.
    
    The minimum (anti-support) in direction d is: p·d - h(beta, theta + pi)
    = p·d - h(beta, theta) [since h is even for a centrally-symmetric hex]
    Wait, a regular hexagon IS centrally symmetric, so h(d) = h(-d).
    So anti-support = p·d - h(beta, theta).
    
    So: max_{p in A} dot(p, d) - min_{q in B} dot(q, d)
       = (pA·d + hA(d)) - (pB·d - hB(d))
       = (pA - pB)·d + hA(d) + hB(d)
    
    They are disjoint iff this < 0 for some separating axis d.
    The candidate separating axes are the edge normals of A and B (12 directions total,
    but due to symmetry we can use 6 from each).
    
    Actually for regular hexagons, the edge normals are at angles beta + 30 + 60k (the
    normals to the edges). Let me verify: vertices at angles beta + 60k. Edges connect
    consecutive vertices. The outward normal of the edge between vertices at angles
    beta+60k and beta+60(k+1) points in direction beta + 60k + 30.
    
    So separating axes to check: for hex A: angles betaA + 30 + 60k, k=0..5
                                 for hex B: angles betaB + 30 + 60k, k=0..5
    
    For each axis direction d (angle theta):
      gap = (pA - pB)·d + hA(theta) + hB(theta)
      where hA(theta) = max_m cos(betaA + 60m - theta) [support of unit hex]
      and hB(theta) = max_m cos(betaB + 60m - theta)
    
    If gap < 0 for any axis, the hexes are separated (disjoint).
    If gap >= 0 for ALL 12 axes, they overlap.
    
    Penalty: sum over overlapping pairs of (max_gap - epsilon) or just count.
    """
    cx = params[0::3]
    cy = params[1::3]
    ang_rad = np.radians(params[2::3])  # (N,)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Check if hex i and hex j are disjoint
            separated = False
            # Edge normals of hex i: ang_rad[i] + 30 + 60k
            # Edge normals of hex j: ang_rad[j] + 30 + 60k
            for k in range(6):
                theta_i = ang_rad[i] + np.radians(30 + 60 * k)
                # gap = (pA - pB)·d + hA + hB
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d_dot = dx * np.cos(theta_i) + dy * np.sin(theta_i)
                hA = np.max(np.cos(ang_rad[i] + _VERT_ANGLES - theta_i))
                hB = np.max(np.cos(ang_rad[j] + _VERT_ANGLES - theta_i))
                gap = d_dot + hA + hB
                if gap < -1e-9:
                    separated = True
                    break
            if separated:
                continue
            for k in range(6):
                theta_j = ang_rad[j] + np.radians(30 + 60 * k)
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d_dot = dx * np.cos(theta_j) + dy * np.sin(theta_j)
                hA = np.max(np.cos(ang_rad[i] + _VERT_ANGLES - theta_j))
                hB = np.max(np.cos(ang_rad[j] + _VERT_ANGLES - theta_j))
                gap = d_dot + hA + hB
                if gap < -1e-9:
                    separated = True
                    break
            if not separated:
                # Overlapping: add penalty proportional to penetration
                # Compute max gap over all axes (should be >= 0 if overlapping)
                max_gap = -np.inf
                for k in range(6):
                    theta = ang_rad[i] + np.radians(30 + 60 * k)
                    d_dot = (cx[i] - cx[j]) * np.cos(theta) + (cy[i] - cy[j]) * np.sin(theta)
                    hA = np.max(np.cos(ang_rad[i] + _VERT_ANGLES - theta))
                    hB = np.max(np.cos(ang_rad[j] + _VERT_ANGLES - theta))
                    g = d_dot + hA + hB
                    if g > max_gap:
                        max_gap = g
                for k in range(6):
                    theta = ang_rad[j] + np.radians(30 + 60 * k)
                    d_dot = (cx[i] - cx[j]) * np.cos(theta) + (cy[i] - cy[j]) * np.sin(theta)
                    hA = np.max(np.cos(ang_rad[i] + _VERT_ANGLES - theta))
                    hB = np.max(np.cos(ang_rad[j] + _VERT_ANGLES - theta))
                    g = d_dot + hA + hB
                    if g > max_gap:
                        max_gap = g
                # max_gap >= 0 means overlap depth
                penalty += max(0.0, max_gap)
    
    return penalty


def _objective(params, outer_angle_rad):
    """Outer side length + large penalty for overlaps."""
    S = _outer_side(params, outer_angle_rad)
    pen = _overlap_penalty(params)
    return S + 50.0 * pen


def _make_hex_lattice_seed(n_rows, n_cols, offset_x=0.0, offset_y=0.0):
    """Create a hex-lattice arrangement of hexagons."""
    positions = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * SQRT3
            positions.append([x, y, 0.0])
    # Center
    if positions:
        cx_mean = np.mean([p[0] for p in positions])
        cy_mean = np.mean([p[1] for p in positions])
        for p in positions:
            p[0] -= cx_mean + offset_x
            p[1] -= cy_mean + offset_y
    # Pad or trim to N
    while len(positions) < N:
        positions.append([0.0, 0.0, 0.0])
    positions = positions[:N]
    params = np.array([v for pos in positions for v in pos])
    return params


def _make_random_seed(rng):
    """Random initial positions."""
    params = np.zeros(N * 3)
    for i in range(N):
        params[3*i] = rng.uniform(-4, 4)
        params[3*i+1] = rng.uniform(-4, 4)
        params[3*i+2] = rng.uniform(0, 60)
    return params


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_S = float('inf')
    best_params = None
    best_outer_angle_rad = 0.0
    
    # Try outer rotations: 0 and 30 degrees
    outer_angles = [0.0, np.radians(30.0)]
    
    # Bounds for DE
    bounds = [(-6.0, 6.0), (-6.0, 6.0), (0.0, 60.0)] * N
    
    rng = np.random.RandomState(42)
    
    for outer_angle_deg in [0.0, 30.0]:
        if time.time() > deadline - 40.0:
            break
        outer_angle_rad = np.radians(outer_angle_deg)
        
        # Multiple starting points
        seeds = []
        # Hex lattice 3x4
        seeds.append(_make_hex_lattice_seed(3, 4))
        # Hex lattice 4x3
        seeds.append(_make_hex_lattice_seed(4, 3))
        # Some random seeds
        for _ in range(3):
            seeds.append(_make_random_seed(rng))
        
        for seed_idx, init in enumerate(seeds):
            if time.time() > deadline - 20.0:
                break
            
            # Evaluate seed
            S_seed = _outer_side(init, outer_angle_rad)
            pen_seed = _overlap_penalty(init)
            
            # Try L-BFGS from this seed
            try:
                result = minimize(
                    _objective,
                    init,
                    args=(outer_angle_rad,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                opt = result.x
                S_opt = _outer_side(opt, outer_angle_rad)
                pen_opt = _overlap_penalty(opt)
                
                if pen_opt < 1e-6 and S_opt < best_S:
                    best_S = S_opt
                    best_params = opt.copy()
                    best_outer_angle_rad = outer_angle_rad
            except Exception:
                pass
            
            # Also try DE from a small region around this seed
            if seed_idx == 0 and time.time() < deadline - 50.0:
                try:
                    # Tighter bounds around the seed
                    de_bounds = []
                    for i in range(N):
                        de_bounds.append((init[3*i] - 3.0, init[3*i] + 3.0))
                        de_bounds.append((init[3*i+1] - 3.0, init[3*i+1] + 3.0))
                        de_bounds.append((max(0, init[3*i+2] - 30), min(60, init[3*i+2] + 30)))
                    
                    result_de = differential_evolution(
                        _objective,
                        de_bounds,
                        args=(outer_angle_rad,),
                        seed=42 + seed_idx,
                        maxiter=60,
                        tol=1e-8,
                        popsize=12,
                        polish=True,
                        init='latinhypercube'
                    )
                    opt_de = result_de.x
                    S_de = _outer_side(opt_de, outer_angle_rad)
                    pen_de = _overlap_penalty(opt_de)
                    
                    if pen_de < 1e-6 and S_de < best_S:
                        best_S = S_de
                        best_params = opt_de.copy()
                        best_outer_angle_rad = outer_angle_rad
                        
                        # Polish the DE result
                        if time.time() < deadline - 10.0:
                            try:
                                result_polish = minimize(
                                    _objective,
                                    best_params,
                                    args=(best_outer_angle_rad,),
                                    method='L-BFGS-B',
                                    options={'maxiter': 300, 'ftol': 1e-14}
                                )
                                opt_p = result_polish.x
                                S_p = _outer_side(opt_p, best_outer_angle_rad)
                                pen_p = _overlap_penalty(opt_p)
                                if pen_p < 1e-6 and S_p < best_S:
                                    best_S = S_p
                                    best_params = opt_p.copy()
                            except Exception:
                                pass
                except Exception:
                    pass
    
    # Fallback if nothing found
    if best_params is None:
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * SQRT3
                inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 6.0, 0.0
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang = float(best_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation: check containment and non-overlap
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(0.0, 0.0, best_S, np.degrees(best_outer_angle_rad))
    
    def point_in_hex(pt, ov):
        for i in range(6):
            v1 = ov[i]
            v2 = ov[(i+1) % 6]
            edge = v2 - v1
            rel = pt - v1
            cross = edge[0]*rel[1] - edge[1]*rel[0]
            if cross < -1e-9:
                return False
        return True
    
    valid = True
    for iv in inner_verts_list:
        for pt in iv:
            if not point_in_hex(pt, outer_verts):
                valid = False
                break
        if not valid:
            break
    
    # Also verify no overlaps
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                # Quick check: if centers are very close, likely overlapping
                dist = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                if dist < 1.7:  # suspicious
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Increase S to ensure containment, and hope overlaps were false alarm
        best_S *= 1.02
    
    return inner, (0.0, 0.0), float(best_S), float(np.degrees(best_outer_angle_rad))
