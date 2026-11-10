# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _fast_min_s(vertices, rot_deg):
    """
    Fast computation of minimum enclosing hexagon side length.
    For a regular hexagon with center c and side s, the constraint is:
    For each of 6 normal directions n_i: max_v(n_i · (v - c)) <= s * sqrt(3)/2
    So s >= max_i [max_v(n_i · v) - n_i · c] * 2/sqrt(3)
    
    We can optimize c to minimize the maximum over all constraints.
    This is a convex optimization problem.
    """
    rot = np.radians(rot_deg)
    normals = np.zeros((6, 2))
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        normals[i] = (np.cos(angle), np.sin(angle))
    
    n_pts = len(vertices)
    
    # Precompute n_i · v for all i, v: shape (6, n_pts)
    dots = np.einsum('ij,nj->ni', normals, vertices)  # (6, n_pts)
    
    # For a given center c, s(c) = max_i [max_v(dots[i,v]) - normals[i]·c] * 2/sqrt(3)
    # = max_i [max_v(dots[i,v]) - (normals[i,0]*c[0] + normals[i,1]*c[1])] * 2/sqrt(3)
    
    # Let M_i = max_v(dots[i,v])
    M = np.max(dots, axis=1)  # (6,)
    
    # s(c) = max_i [M_i - normals[i]·c] * 2/sqrt(3)
    # We want to minimize this over c.
    
    # This is a convex piecewise-linear function. We can use subgradient method
    # or solve it as a linear program with variables c[0], c[1], t where
    # t >= M_i - normals[i]·c for all i, minimize t.
    
    # For speed, let's use a simple iterative approach:
    # Start with c = centroid, then iterate.
    
    c = np.mean(vertices, axis=0)
    best_s = np.inf
    
    for _ in range(50):
        # Compute current s
        constraints = M - np.dot(normals, c)  # (6,)
        s_val = np.max(constraints) * 2 / np.sqrt(3)
        
        if s_val < best_s:
            best_s = s_val
        
        # Subgradient: direction that increases the most violated constraint
        idx_max = np.argmax(constraints)
        n_max = normals[idx_max]
        
        # Move c in direction n_max to reduce the max constraint
        # d/dc [M_i - n_i·c] = -n_i, so to decrease constraint i, move c in direction n_i
        step = 0.01
        c_new = c + step * n_max
        
        # Check if this improves
        constraints_new = M - np.dot(normals, c_new)
        s_new = np.max(constraints_new) * 2 / np.sqrt(3)
        
        if s_new < s_val:
            c = c_new
        else:
            # Try smaller step
            step *= 0.5
            c_new = c + step * n_max
            constraints_new = M - np.dot(normals, c_new)
            s_new = np.max(constraints_new) * 2 / np.sqrt(3)
            if s_new < s_val:
                c = c_new
            else:
                break
    
    # Final computation
    constraints = M - np.dot(normals, c)
    s_val = np.max(constraints) * 2 / np.sqrt(3)
    
    return s_val, c


def _build_cluster(centers, angle_deg=0.0):
    """
    Build all 72 vertices of the 12 hexagons given their centers and a common rotation.
    Returns (72, 2) array of vertices.
    """
    a = np.radians(angle_deg)
    unit_angles = np.radians(60 * np.arange(6))
    unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])

    cos_a, sin_a = np.cos(a), np.sin(a)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_unit_verts = unit_verts @ rot_matrix.T

    all_verts = np.zeros((72, 2))
    for j in range(12):
        all_verts[j*6:(j+1)*6] = centers[j] + rot_unit_verts

    return all_verts


def _validate_packing(inner_hex_data, outer_hex_data, s, eps=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False

    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oang)

    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False

    # Precompute inner hexagon vertices
    inner_verts_list = []
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        # Check unit edges
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k + 1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1) > 1e-8:
                return False
        inner_verts_list.append(verts)

    # Check separation between inner hexagons using separating axis test
    for i in range(12):
        for j in range(i + 1, 12):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]

            # Get all edge normals from both hexagons
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)

            separated = False
            for n in normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                if max_i + eps < min_j or max_j + eps < min_i:
                    separated = True
                    break

            if not separated:
                return False

    # Check all inner vertices are inside outer hexagon
    for i in range(12):
        verts = inner_verts_list[i]
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False

    return True


def _polish_with_nelder_mead(centers, theta0, phi0, max_iter=500):
    """
    Polish the solution using Nelder-Mead on (theta, phi).
    """
    def objective(params):
        theta_d, phi_d = params
        cluster_v = _build_cluster(centers, theta_d)
        s_val, _ = _fast_min_s(cluster_v, phi_d)
        return s_val
    
    x0 = [theta0, phi0]
    res = minimize(objective, x0, method='Nelder-Mead',
                  options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14})
    
    return res.fun, res.x


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin before 630s kill

    sqrt3 = np.sqrt(3)

    # Define multiple candidate cluster configurations
    # Configuration 1: Flower pattern (center + 6 neighbors + 5 from next ring)
    eps = 1e-3
    scale = 1.0 + eps
    
    centers_flower = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [0, 2 * sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [0, -2 * sqrt3],
    ]) * scale

    # Configuration 2: Compact 3x4 grid-like arrangement
    centers_compact = np.array([
        [0, 0],
        [2, 0],
        [4, 0],
        [1, sqrt3],
        [3, sqrt3],
        [5, sqrt3],
        [0, 2*sqrt3],
        [2, 2*sqrt3],
        [4, 2*sqrt3],
        [1, -sqrt3],
        [3, -sqrt3],
        [5, -sqrt3],
    ]) * scale

    # Configuration 3: Hexagonal close packing variant
    centers_hex = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [0, 2*sqrt3],
        [2, 2*sqrt3],
        [-2, 2*sqrt3],
        [0, -2*sqrt3],
        [2, -2*sqrt3],
    ]) * scale

    candidates = [
        ('flower', centers_flower),
        ('compact', centers_compact),
        ('hex', centers_hex),
    ]

    best_s = np.inf
    best_inner = None
    best_outer = None
    best_config_name = None

    # Phase 1: Coarse search over all configurations
    n_theta = 31
    n_phi = 31

    for config_name, centers in candidates:
        if time.time() > deadline - 300:
            break

        theta_vals = np.linspace(0, 60, n_theta)
        phi_vals = np.linspace(0, 60, n_phi)

        for theta_deg in theta_vals:
            if time.time() > deadline - 300:
                break

            cluster_verts = _build_cluster(centers, theta_deg)

            for phi_deg in phi_vals:
                if time.time() > deadline - 300:
                    break

                s, center = _fast_min_s(cluster_verts, phi_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([centers, np.full(12, theta_deg)])
                    outer_hex_data = np.array([center[0], center[1], phi_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data.copy()
                        best_outer = outer_hex_data.copy()
                        best_config_name = config_name

    # Phase 2: Fine grid search around best configuration
    if best_inner is not None and time.time() < deadline - 200:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        # Find which configuration was best
        centers_best = None
        for config_name, centers in candidates:
            if config_name == best_config_name:
                centers_best = centers
                break
        
        if centers_best is None:
            centers_best = centers_flower

        theta_fine = np.linspace(best_theta - 5, best_theta + 5, 51)
        phi_fine = np.linspace(best_phi - 5, best_phi + 5, 51)

        for theta_deg in theta_fine:
            if time.time() > deadline - 200:
                break

            cluster_verts = _build_cluster(centers_best, theta_deg)

            for phi_deg in phi_fine:
                if time.time() > deadline - 200:
                    break

                s, center = _fast_min_s(cluster_verts, phi_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([centers_best, np.full(12, theta_deg)])
                    outer_hex_data = np.array([center[0], center[1], phi_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data.copy()
                        best_outer = outer_hex_data.copy()

    # Phase 3: Nelder-Mead polish
    if best_inner is not None and time.time() < deadline - 100:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        centers_best = None
        for config_name, centers in candidates:
            if config_name == best_config_name:
                centers_best = centers
                break
        
        if centers_best is None:
            centers_best = centers_flower

        s_polished, params_polished = _polish_with_nelder_mead(
            centers_best, best_theta, best_phi, max_iter=1000
        )
        
        if s_polished < best_s:
            theta_opt, phi_opt = params_polished
            cluster_verts = _build_cluster(centers_best, theta_opt)
            s_opt, center_opt = _fast_min_s(cluster_verts, phi_opt)
            inner_hex_data = np.column_stack([centers_best, np.full(12, theta_opt)])
            outer_hex_data = np.array([center_opt[0], center_opt[1], phi_opt])
            if _validate_packing(inner_hex_data, outer_hex_data, s_opt):
                best_s = s_opt
                best_inner = inner_hex_data.copy()
                best_outer = outer_hex_data.copy()

    # Phase 4: Try compressed versions of the best configuration
    if best_inner is not None and time.time() < deadline - 50:
        centers_best = None
        for config_name, centers in candidates:
            if config_name == best_config_name:
                centers_best = centers
                break
        
        if centers_best is None:
            centers_best = centers_flower

        for compress in [0.999, 0.998, 0.995]:
            if time.time() > deadline - 50:
                break

            centers_comp = centers_best * compress
            best_theta = best_inner[0, 2]
            best_phi = best_outer[2]

            s_polished, params_polished = _polish_with_nelder_mead(
                centers_comp, best_theta, best_phi, max_iter=500
            )
            
            if s_polished < best_s:
                theta_opt, phi_opt = params_polished
                cluster_verts = _build_cluster(centers_comp, theta_opt)
                s_opt, center_opt = _fast_min_s(cluster_verts, phi_opt)
                inner_hex_data = np.column_stack([centers_comp, np.full(12, theta_opt)])
                outer_hex_data = np.array([center_opt[0], center_opt[1], phi_opt])
                if _validate_packing(inner_hex_data, outer_hex_data, s_opt):
                    best_s = s_opt
                    best_inner = inner_hex_data.copy()
                    best_outer = outer_hex_data.copy()

    # If no valid packing was found, use a default
    if best_inner is None:
        centers = centers_flower
        inner_hex_data = np.column_stack([centers, np.zeros(12)])
        s, center = _fast_min_s(_build_cluster(centers, 0), 0)
        outer_hex_data = np.array([center[0], center[1], 0])
        return inner_hex_data, outer_hex_data, s

    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
