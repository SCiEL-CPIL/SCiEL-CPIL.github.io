# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))

def _hex_edge_normals(vertices):
    """Return outward normals of the 6 edges of a hexagon given its 6 vertices."""
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normals[i] = normal / norm
    return normals

def _separation_check(v1, v2):
    """Check if two hexagons are separated (SAT). Returns True if separated."""
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _is_inside_outer(inner_vertices, outer_vertices):
    """Check if all vertices of inner hex are inside outer hex."""
    n_out = len(outer_vertices)
    for v in inner_vertices:
        inside = True
        for i in range(n_out):
            p1 = outer_vertices[i]
            p2 = outer_vertices[(i + 1) % n_out]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                inside = False
                break
        if not inside:
            return False
    return True

def _min_separation(v1, v2):
    """Compute the minimum separation between two hexagons. Positive if separated, negative if overlapping."""
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    min_sep = np.inf
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            sep = -min(max1 - min2, max2 - min1)
        min_sep = min(min_sep, sep)
    return min_sep

def _compute_s_rigid(cluster_positions, outer_center, outer_angle_deg, inner_angles=None):
    """Compute minimum s of outer hexagon containing all inner hexagons in rigid cluster."""
    outer_center = np.array(outer_center)
    inner_centers = np.array(cluster_positions)
    if inner_angles is None:
        inner_angles = np.zeros(len(inner_centers))
    else:
        inner_angles = np.array(inner_angles)
    
    shifted = inner_centers - outer_center
    a = np.radians(outer_angle_deg)
    normal_angles = a + np.radians(60) * np.arange(6) + np.radians(30)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    all_vertices = []
    for i in range(len(inner_centers)):
        cx, cy = shifted[i]
        ang = inner_angles[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    projections = all_vertices @ normals.T
    max_proj = projections.max()
    s = 2 * max_proj / np.sqrt(3)
    return s

def _build_honeycomb_cluster(rows):
    """Build a honeycomb cluster from row specifications.
    rows: list of (count, y_offset) where y_offset is in units of sqrt(3)/2
    Returns (12, 2) array of center positions.
    """
    sqrt3 = np.sqrt(3)
    centers = []
    y = 0
    for count, y_off in rows:
        actual_y = y_off * sqrt3 / 2
        x_start = -(count - 1) * 0.5
        for i in range(count):
            centers.append([x_start + i * 1.0, actual_y])
    return np.array(centers)

def _rotate_points(points, angle_deg):
    """Rotate points around origin by angle_deg."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T

def _compute_s_for_rotation(cluster_rot, outer_angle_deg):
    """Compute s for a given cluster rotation and outer angle, with outer center at origin."""
    outer_center = np.array([0.0, 0.0])
    return _compute_s_rigid(cluster_rot, outer_center, outer_angle_deg)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Build the 3-4-3-2 honeycomb cluster
    # Row 0: 3 hexagons at y=0
    # Row 1: 4 hexagons at y=sqrt(3)/2
    # Row 2: 3 hexagons at y=sqrt(3)
    # Row 3: 2 hexagons at y=3*sqrt(3)/2
    rows = [(3, 0), (4, 1), (3, 2), (2, 3)]
    base_cluster = _build_honeycomb_cluster(rows)
    
    # Center the cluster at origin
    centroid = np.mean(base_cluster, axis=0)
    base_cluster = base_cluster - centroid
    
    # We'll optimize: cluster rotation (1 var) and outer angle (1 var)
    # Outer center is fixed at origin by symmetry
    
    best_s = np.inf
    best_cluster_rot = None
    best_outer_angle = 0
    
    # Strategy 1: Coarse grid search over cluster rotation and outer angle
    # The problem has 6-fold symmetry, so we only need to search 0 to 60 degrees
    # But cluster and outer angles are independent, so search both in [0, 60)
    
    # First, try a grid of cluster rotations with optimal outer angle for each
    n_cluster_steps = 120  # 0.5 degree steps
    n_outer_steps = 120
    
    # Pre-compute: for each cluster rotation, find best outer angle
    # To save time, use a coarser outer search first, then refine
    
    # Phase 1: Coarse search
    coarse_cluster = np.linspace(0, 60, 30, endpoint=False)
    coarse_outer = np.linspace(0, 60, 30, endpoint=False)
    
    for cr in coarse_cluster:
        if time.time() > deadline - 60:
            break
        cluster_rot = _rotate_points(base_cluster, cr)
        for oa in coarse_outer:
            s = _compute_s_for_rotation(cluster_rot, oa)
            if s < best_s:
                best_s = s
                best_cluster_rot = cluster_rot.copy()
                best_outer_angle = oa
    
    # Phase 2: Refine around best found
    if time.time() < deadline - 30:
        # Local refinement using Nelder-Mead on 2 variables: cluster_rot_angle, outer_angle
        def objective(params):
            cr_angle = params[0]
            oa_angle = params[1]
            cluster_rot = _rotate_points(base_cluster, cr_angle)
            return _compute_s_for_rotation(cluster_rot, oa_angle)
        
        # Multiple restarts around the best
        restarts = [(best_cluster_rot, best_outer_angle)]
        # Add some nearby points
        for dcr in [-2, -1, 1, 2]:
            for doa in [-2, -1, 1, 2]:
                restarts.append((best_cluster_rot, best_outer_angle + dcr * 1.0))
        
        for r in range(min(6, len(restarts))):
            if time.time() > deadline - 15:
                break
            cr_init = best_cluster_rot
            oa_init = best_outer_angle + (r - 2) * 1.0
            try:
                res = minimize(objective, [0, oa_init], method='Nelder-Mead',
                              options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200, 'maxfev': 500})
                if res.fun < best_s:
                    best_s = res.fun
                    cr_opt = res.x[0]
                    oa_opt = res.x[1]
                    best_cluster_rot = _rotate_points(base_cluster, cr_opt)
                    best_outer_angle = oa_opt
            except Exception:
                pass
    
    # Phase 3: Even finer grid if time permits
    if time.time() < deadline - 20 and best_s > 0:
        fine_cluster = np.linspace(0, 60, 240, endpoint=False)
        for cr in fine_cluster:
            if time.time() > deadline - 10:
                break
            cluster_rot = _rotate_points(base_cluster, cr)
            # For this cluster rotation, do a fine outer search
            fine_outer = np.linspace(0, 60, 240, endpoint=False)
            for oa in fine_outer:
                s = _compute_s_for_rotation(cluster_rot, oa)
                if s < best_s:
                    best_s = s
                    best_cluster_rot = cluster_rot.copy()
                    best_outer_angle = oa
    
    # Final refinement with Nelder-Mead
    if time.time() < deadline - 5:
        def objective_final(params):
            cr_angle = params[0]
            oa_angle = params[1]
            cluster_rot = _rotate_points(base_cluster, cr_angle)
            return _compute_s_for_rotation(cluster_rot, oa_angle)
        
        try:
            res = minimize(objective_final, [0, best_outer_angle], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 1000})
            if res.fun < best_s:
                best_s = res.fun
                cr_opt = res.x[0]
                oa_opt = res.x[1]
                best_cluster_rot = _rotate_points(base_cluster, cr_opt)
                best_outer_angle = oa_opt
        except Exception:
            pass
    
    # Construct the final answer
    # Inner positions are best_cluster_rot (already centered at origin)
    # Outer center is origin, outer angle is best_outer_angle
    # Inner angles are all 0 (aligned with cluster)
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_cluster_rot[i, 0]
        inner_hex_data[i, 1] = best_cluster_rot[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    # Verify validity
    valid = True
    # Check separation between all pairs
    for i in range(12):
        for j in range(i + 1, 12):
            v1 = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1, 0)
            v2 = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1, 0)
            if not _separation_check(v1, v2):
                valid = False
                break
        if not valid:
            break
    
    # Check all inner vertices inside outer
    if valid:
        outer_verts = _hex_vertices(0, 0, best_s, best_outer_angle)
        for i in range(12):
            inner_verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1, 0)
            if not _is_inside_outer(inner_verts, outer_verts):
                valid = False
                break
    
    if not valid:
        # Fallback: use the seed solution
        init_pos = np.array([
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.17],
            [1.25, 2.17],
            [-1.25, -2.17],
            [1.25, -2.17],
            [-3.75, 2.17],
            [3.75, 2.17],
            [-3.75, -2.17],
            [3.75, -2.17],
            [0, -4]
        ])
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i, 0] = init_pos[i, 0]
            inner_hex_data[i, 1] = init_pos[i, 1]
            inner_hex_data[i, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        best_s = 8.0
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
