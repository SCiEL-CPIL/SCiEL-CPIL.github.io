"""Semi-infinite packing: linearized non-overlap constraints via 120-angle SAT projection."""
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 12
SQRT3 = math.sqrt(3.0)

# Unit hexagon vertices (circumradius = 1 = side length)
_UNIT_VERTS = np.array([
    [1.0, 0.0],
    [0.5, SQRT3 / 2.0],
    [-0.5, SQRT3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -SQRT3 / 2.0],
    [0.5, -SQRT3 / 2.0],
])

# 120 critical angles (every 3 degrees) for non-overlap checking
_CRITICAL_ANGLES = np.array([math.radians(3.0 * k) for k in range(120)])
_CRITICAL_NORMALS = np.column_stack([np.cos(_CRITICAL_ANGLES), np.sin(_CRITICAL_ANGLES)])

# Precompute support values for unit hexagon at all critical angles
# s(theta) = max over vertices v of v . n(theta)
_SV = _UNIT_VERTS @ _CRITICAL_NORMALS.T  # (6, 120)
_SV = _SV.max(axis=0)  # (120,)


def _hex_vertices(cx, cy, angle_deg):
    a = math.radians(angle_deg)
    angles = a + np.arange(6) * (math.pi / 3.0)
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _separation_sat(v1, v2):
    """Continuous separation measure between two convex hexagons via SAT on all 12 edges."""
    min_sep = float('inf')
    for verts in (v1, v2):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            nrm = np.array([-edge[1], edge[0]])
            nn = np.linalg.norm(nrm) + 1e-12
            nrm = nrm / nn
            p1 = v1 @ nrm
            p2 = v2 @ nrm
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                sep = -min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
    return min_sep


def _all_separations(verts_list):
    seps = []
    for i in range(N):
        for j in range(i + 1, N):
            seps.append(_separation_sat(verts_list[i], verts_list[j]))
    return seps


def _support_unit(nvec):
    """Max dot product of unit hexagon vertices with direction nvec (2-vector)."""
    return float(np.max(_UNIT_VERTS @ nvec))


def _outer_side_support(positions, alpha_deg, beta_deg):
    """
    Minimal outer hexagon side length to contain N unit hexagons at `positions`,
    all rotated by alpha_deg, inside outer hexagon rotated by beta_deg (centered at origin).
    """
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    R_neg_a = np.array([[ca, sa], [-sa, ca]])  # R(-alpha)

    # Outer normals at angle beta
    beta = math.radians(beta_deg)
    cb, sb = math.cos(beta), math.sin(beta)
    outer_angles = beta + np.array([math.radians(30.0 + 60.0 * k) for k in range(6)])
    outer_normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])

    # Rotate outer normals by -alpha to get direction in inner hexagon frame
    n_rot = outer_normals @ R_neg_a.T  # (6,2)
    h = np.array([_support_unit(n) for n in n_rot])  # (6,)

    # Project positions onto outer normals
    proj = positions @ outer_normals.T + h[None, :]
    max_proj = proj.max(axis=0)
    return float(2.0 * np.max(max_proj) / SQRT3)


def _build_axial_positions(scale=2.0):
    """Generate several lattice topologies as centered Cartesian positions."""
    topologies = []

    # Helper: axial (q,r) -> cartesian
    def axial_to_cart(coords):
        pts = []
        for q, r in coords:
            x = scale * (q + r / 2.0)
            y = scale * (r * SQRT3 / 2.0)
            pts.append((x, y))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        return np.array([(x - cx, y - cy) for x, y in pts])

    # 3x4 grid
    topologies.append(axial_to_cart([(q, r) for q in range(3) for r in range(4)]))
    # 4x3 grid
    topologies.append(axial_to_cart([(q, r) for q in range(4) for r in range(3)]))
    # 2x6 grid
    topologies.append(axial_to_cart([(q, r) for q in range(2) for r in range(6)]))
    # 6x2 grid
    topologies.append(axial_to_cart([(q, r) for q in range(6) for r in range(2)]))

    # Hexagonal cluster: center + ring1 + 5 from ring2
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    base = [(0, 0)] + ring1
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            d = (abs(q) + abs(r) + abs(q + r)) // 2
            if d == 2:
                ring2.append((q, r))

    def ang(p):
        x = p[0] + p[1] / 2.0
        y = p[1] * SQRT3 / 2.0
        return math.atan2(y, x)

    ring2_sorted = sorted(ring2, key=ang)
    picks = [ring2_sorted[i] for i in range(0, len(ring2_sorted), 2)][:5]
    while len(picks) < 5:
        picks.append(ring2_sorted[len(picks) % len(ring2_sorted)])
    topologies.append(axial_to_cart(base + picks))

    # Cross shape
    cross_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, -2)
    ]
    if len(set(cross_coords)) == 12:
        topologies.append(axial_to_cart(cross_coords))

    # Staggered 2x6
    staggered = []
    for r in range(6):
        offset = 0 if r % 2 == 0 else 1
        for q in range(2):
            staggered.append((q + offset, r))
    topologies.append(axial_to_cart(staggered))

    return topologies


def _linprog_nonoverlap_check(positions, alpha_deg, n_angles=120):
    """
    Check non-overlap using linearized SAT constraints.
    For each pair (i,j) and each critical angle theta, verify that
    the projection intervals are disjoint OR use big-M to handle separation direction.
    
    Returns True if all pairs are separated (no overlap detected).
    """
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, sa], [-sa, ca]])  # R(-alpha)

    # Precompute rotated hexagon support in all critical directions
    # For a hexagon at position p with rotation alpha, its support in direction n is:
    #   p . n + s_unit(R(-alpha) n)
    # where s_unit is the support of the unit hexagon.

    # For each critical angle theta, n = [cos(theta), sin(theta)]
    # Rotated direction: n' = R(-alpha) n
    # We need: for all i != j, there exists a direction where they are separated
    # But SAT says: convex polygons are disjoint iff there exists a separating axis.
    # We check on a finite set of axes (120 angles + 12 edge normals per hexagon).

    # For efficiency, check the 120 critical angles
    # For each angle, compute projections of all N hexagons
    # If for EVERY angle, some pair overlaps, then they might still be disjoint (SAT not complete)
    # But if for SOME angle, all pairs are separated, that's strong evidence.
    # Actually, we need: for each pair, AT LEAST ONE angle shows separation.

    # More practical: for each pair, check if there's an angle where they're separated
    # This is what SAT does.

    # Simplified: compute projections at all 120 angles for each hexagon
    # proj[i, k] = positions[i] . n_k + s_unit(R(-alpha) n_k)
    # where n_k = [cos(theta_k), sin(theta_k)]

    n_crit = _CRITICAL_NORMALS  # (120, 2)
    n_rot = n_crit @ R.T  # (120, 2)
    # Support of unit hex in each rotated direction
    h = np.array([_support_unit(n) for n in n_rot])  # (120,)

    # Projections: (N, 120)
    proj = positions @ n_crit.T + h[None, :]  # (N, 120)

    # For each pair (i,j), check if there exists an angle k where
    # proj[i,k] + 1 < proj[j,k] OR proj[j,k] + 1 < proj[i,k]
    # (the +1 accounts for the hexagon's extent in that direction, but we already included support)
    # Actually, proj[i,k] is the support value (max extent), so the interval is [proj[i,k] - 2*s, proj[i,k]]
    # No wait, support is the max, so the interval is [min, max] where max = proj[i,k]
    # The min would be the support in the opposite direction.
    
    # Let me reconsider. The projection of a convex polygon onto direction n gives an interval [min, max].
    # max = p . n + s_unit(R(-alpha) n) where s_unit is max of vertex dot n_rot
    # min = p . n - s_unit(R(-alpha)(-n)) = p . n - s_unit(R(-alpha) n + pi)
    # For a centrally symmetric shape like a hexagon, s_unit(n) = s_unit(-n), so
    # min = p . n - s_unit(R(-alpha) n)
    # So the interval is [p . n - h_k, p . n + h_k] where h_k = s_unit(R(-alpha) n_k)

    # So for separation in direction k:
    # (p_i . n_k + h_k) < (p_j . n_k - h_k)  =>  (p_i - p_j) . n_k < -2*h_k
    # OR (p_j . n_k + h_k) < (p_i . n_k - h_k)  =>  (p_j - p_i) . n_k < -2*h_k
    # i.e., |p_i - p_j| projected onto n_k has magnitude > 2*h_k in the separating direction.

    # Check: for each pair, is there a k where separation holds?
    for i in range(N):
        for j in range(i + 1, N):
            diff = positions[i] - positions[j]  # (2,)
            # diff @ n_crit.T gives (120,) dot products
            dots = diff @ n_crit.T  # (120,)
            # Separation if dots[k] > 2*h[k] or dots[k] < -2*h[k]
            # i.e., |dots[k]| > 2*h[k]
            if not np.any(np.abs(dots) > 2.0 * h + 1e-9):
                return False  # No separating axis found for this pair
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    topologies = _build_axial_positions()

    best_side = float('inf')
    best_positions = None
    best_alpha = 0.0
    best_beta = 0.0

    # Phase 1: Grid search over topologies and (alpha, beta)
    for topo in topologies:
        if time.time() > deadline - 30.0:
            break
        positions = topo.copy()
        best_local = float('inf')
        best_local_a = 0.0
        best_local_b = 0.0

        # Coarse grid: alpha in [0,60), beta in [0,60), step 3 degrees
        for a in np.arange(0.0, 60.0, 3.0):
            if time.time() > deadline - 30.0:
                break
            for b in np.arange(0.0, 60.0, 3.0):
                s = _outer_side_support(positions, a, b)
                if s < best_local:
                    best_local = s
                    best_local_a = a
                    best_local_b = b

        if best_local < best_side:
            best_side = best_local
            best_positions = positions.copy()
            best_alpha = best_local_a
            best_beta = best_local_b

    if best_positions is None:
        best_positions = topologies[0].copy()
        best_alpha, best_beta = 0.0, 0.0
        best_side = _outer_side_support(best_positions, 0.0, 0.0)

    # Phase 2: Nelder-Mead refinement of (alpha, beta)
    if time.time() < deadline - 40.0:
        pos = best_positions

        def obj_ab(params):
            return _outer_side_support(pos, params[0], params[1])

        try:
            res = minimize(obj_ab, [best_alpha, best_beta], method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_alpha = float(res.x[0]) % 60.0
                best_beta = float(res.x[1]) % 60.0
        except Exception:
            pass

    # Phase 3: Local position optimization with linearized non-overlap via penalty
    if time.time() < deadline - 50.0:
        pos = best_positions.copy()
        alpha = best_alpha

        def penalized_objective(x):
            """x = [x0,y0, x1,y1, ..., x11,y11, alpha, beta] (24+2+1 = 27 params)"""
            positions = x[:24].reshape(N, 2)
            a = float(x[24])
            b = float(x[25])
            side = _outer_side_support(positions, a, b)

            # Penalty for overlap: check all pairs at 120 angles
            # Use the linearized check
            n_crit = _CRITICAL_NORMALS
            ar = math.radians(a)
            ca, sa = math.cos(ar), math.sin(ar)
            R = np.array([[ca, sa], [-sa, ca]])
            n_rot = n_crit @ R.T
            h = np.array([_support_unit(n) for n in n_rot])

            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    diff = positions[i] - positions[j]
                    dots = diff @ n_crit.T
                    # Need |dots[k]| > 2*h[k] for at least one k
                    # Penalty: sum over k of max(0, 2*h[k] - |dots[k]|)^2
                    shortfall = 2.0 * h - np.abs(dots)
                    max_shortfall = shortfall.max()
                    if max_shortfall > 0:
                        penalty += max_shortfall ** 2

            return side + 50.0 * penalty

        x0 = np.zeros(26)
        x0[:24] = pos.flatten()
        x0[24] = alpha
        x0[25] = best_beta

        try:
            res2 = minimize(penalized_objective, x0, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res2.fun < best_side + 1e-6 and res2.fun < 5.0:
                pos_final = res2.x[:24].reshape(N, 2)
                alpha_final = float(res2.x[24]) % 60.0
                beta_final = float(res2.x[25]) % 60.0

                # Verify validity
                verts = [_hex_vertices(pos_final[i, 0], pos_final[i, 1], alpha_final) for i in range(N)]
                seps = _all_separations(verts)
                if min(seps) >= -1e-6:
                    side_final = _outer_side_support(pos_final, alpha_final, beta_final)
                    if side_final < best_side:
                        best_side = side_final
                        best_positions = pos_final
                        best_alpha = alpha_final
                        best_beta = beta_final
        except Exception:
            pass

    # Finalize: recenter and verify
    positions = best_positions
    cx = float(np.mean(positions[:, 0]))
    cy = float(np.mean(positions[:, 1]))
    positions = positions - np.array([cx, cy])

    final_side = _outer_side_support(positions, best_alpha, best_beta)

    # Verify non-overlap with true SAT
    verts = [_hex_vertices(positions[i, 0], positions[i, 1], best_alpha) for i in range(N)]
    seps = _all_separations(verts)
    if min(seps) < -1e-4:
        # Fallback: use the best lattice topology with scale=2.2 for safety gap
        positions = _build_axial_positions(scale=2.2)[0].copy()
        cx = float(np.mean(positions[:, 0]))
        cy = float(np.mean(positions[:, 1]))
        positions = positions - np.array([cx, cy])
        best_alpha = 0.0
        best_beta = 0.0
        verts = [_hex_vertices(positions[i, 0], positions[i, 1], best_alpha) for i in range(N)]
        final_side = _outer_side_support(positions, best_alpha, best_beta)

    inner = [[float(positions[i, 0]), float(positions[i, 1]), float(best_alpha)] for i in range(N)]
    return inner, (0.0, 0.0), float(final_side), float(best_beta)
