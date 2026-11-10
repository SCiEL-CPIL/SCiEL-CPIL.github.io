import numpy as np
import time


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _lattice_to_cartesian(lattice_coords, rotation_angle):
    """
    Convert hexagonal lattice axial coordinates to Cartesian.
    For unit hexagons (side 1), center-to-center distance for edge-adjacent is sqrt(3).
    Axial: x = sqrt(3)*(q + r/2), y = 1.5*r
    """
    q = lattice_coords[:, 0]
    r = lattice_coords[:, 1]
    x = np.sqrt(3.0) * (q + r / 2.0)
    y = 1.5 * r
    theta = np.deg2rad(rotation_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return np.column_stack([x_rot, y_rot])


def _min_s_for_containment(points, outer_center, outer_angle):
    """
    Min side s of regular hexagon (center=outer_center, angle=outer_angle) containing all points.
    Hexagon with center c, side s, angle a has 6 edges. Inward normals at angles a+30+60i.
    Distance from center to each edge = s*sqrt(3)/2.
    For each outward normal n_i, need: max_j dot(p_j - c, n_i) <= s*sqrt(3)/2
    So s >= max_i (max_j dot(p_j - c, n_i)) / (sqrt(3)/2)
    """
    normal_angles = np.deg2rad(outer_angle + 30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    shifted = points - outer_center  # (N, 2)
    projections = shifted @ normals.T  # (N, 6)
    max_proj = np.max(projections, axis=0)  # (6,)
    s_min = np.max(max_proj) / (np.sqrt(3.0) / 2.0)
    return s_min


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, "unit edge length differs"
    # SAT separation check
    def _separated(v1, v2):
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(normal)
            if n < 1e-12:
                continue
            normal /= n
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if np.min(proj2) - np.max(proj1) > 1e-6 or np.min(proj1) - np.max(proj2) > 1e-6:
                return True
        return False

    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner {i},{j} overlap"
    # Containment
    for row in inner:
        hv = _hex_vertices(row[0], row[1], 1.0, row[2])
        for v in hv:
            for k in range(6):
                p1, p2 = ov[k], ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, "inner vertex outside outer"
    return True, ""


def hexagon_packing_12():
    """
    Pack 12 unit hexagons in a regular hexagon.
    Structural approach: fixed compact topology (center + ring1 + 5 ring2),
    exhaustive grid over rotation and outer center, then Nelder-Mead refinement.
    """
    start_time = time.time()
    deadline = start_time + 580.0

    # Compact topology: center + 6 ring-1 + 5 selected ring-2
    lattice_pos = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1],
    ], dtype=float)

    # Precompute: for each rotation, get Cartesian positions (centered at origin)
    # Then we optimize outer_center (ox, oy) and outer_angle to minimize s

    best_s = np.inf
    best_result = None

    def evaluate(theta_deg, ox, oy, outer_angle):
        """Compute s for given rotation of inner, outer center, outer angle."""
        cart = _lattice_to_cartesian(lattice_pos, theta_deg)
        # Center inner configuration at origin for symmetry
        cart_centered = cart - cart.mean(axis=0)
        # All inner hexagons share same orientation = theta_deg
        # Compute min s
        s = _min_s_for_containment(cart_centered, np.array([ox, oy]), outer_angle)
        return s

    # Phase 1: Coarse grid over theta and outer_angle
    # Due to 6-fold symmetry, theta in [0, 60) and outer_angle in [0, 60) suffice
    # But let's be safe and use [0, 360) with step 5 for coarse, then refine
    
    theta_coarse = np.arange(0, 360, 5.0)
    angle_coarse = np.arange(0, 360, 5.0)
    
    # For each (theta, outer_angle), find optimal outer_center by noting
    # that for fixed theta and outer_angle, s(ox,oy) is convex (max of linear functions)
    # We can grid over ox, oy in a reasonable range
    
    # Estimate bounding box for outer center
    # The inner config has extent ~ sqrt(3)*2 = 3.46 in each direction
    # Outer center should be near origin, within ~2 units
    oc_range = np.arange(-2.0, 2.01, 0.5)
    
    for theta in theta_coarse:
        if time.time() > deadline:
            break
        for outer_ang in angle_coarse:
            if time.time() > deadline:
                break
            for ox in oc_range:
                for oy in oc_range:
                    s = evaluate(theta, ox, oy, outer_ang)
                    if s < best_s:
                        best_s = s
                        best_result = (theta, ox, oy, outer_ang)

    # Phase 2: Nelder-Mead refinement around best
    if best_result is not None and time.time() < deadline - 60:
        from scipy.optimize import minimize
        
        def obj(params):
            theta, ox, oy, outer_ang = params
            return evaluate(theta, ox, oy, outer_ang)
        
        x0 = np.array(best_result)
        result = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_s:
            best_s = result.fun
            best_result = tuple(result.x)
        
        # Additional refinement starts
        for perturb in [1.0, -1.0, 2.5, -2.5]:
            if time.time() > deadline - 30:
                break
            x0p = np.array(best_result)
            x0p[0] += perturb
            result2 = minimize(obj, x0p, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result2.fun < best_s:
                best_s = result2.fun
                best_result = tuple(result2.x)
    
    # Phase 3: Fine grid around best if time remains
    if best_result is not None and time.time() < deadline - 30:
        theta_best, ox_best, oy_best, ang_best = best_result
        for dtheta in np.arange(-3, 3.01, 0.5):
            for dang in np.arange(-3, 3.01, 0.5):
                for dox in np.arange(-0.3, 0.31, 0.1):
                    for doy in np.arange(-0.3, 0.31, 0.1):
                        if time.time() > deadline - 10:
                            break
                        s = evaluate(theta_best + dtheta, ox_best + dox, oy_best + doy, ang_best + dang)
                        if s < best_s:
                            best_s = s
                            best_result = (theta_best + dtheta, ox_best + dox, oy_best + doy, ang_best + dang)

    # Build final result
    if best_result is None:
        # Fallback: simple placement
        theta, ox, oy, outer_ang = 0.0, 0.0, 0.0, 0.0
        best_s = 10.0
    else:
        theta, ox, oy, outer_ang = best_result

    cart = _lattice_to_cartesian(lattice_pos, theta)
    cart_centered = cart - cart.mean(axis=0)
    
    inner = np.column_stack([
        cart_centered[:, 0],
        cart_centered[:, 1],
        np.full(12, theta)
    ])
    outer = np.array([ox, oy, outer_ang])
    
    # Add small margin to s for numerical safety
    s_final = best_s + 1e-5
    
    valid, msg = _check_validity(inner, outer, s_final)
    if not valid:
        # Increase s until valid
        for margin in [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1]:
            s_try = best_s + margin
            valid, msg = _check_validity(inner, outer, s_try)
            if valid:
                s_final = s_try
                break
        else:
            s_final = best_s + 0.5

    return inner, outer, s_final
