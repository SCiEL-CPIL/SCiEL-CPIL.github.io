import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"
    
    # Outer hexagon vertices must be distinct
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    # Check unit edge length for inner hexagons
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"unit edge length differs: {edge_len}"
    
    # Check inner-inner separation using SAT
    def _separated(v1, v2):
        """Check if two hexagons are separated (non-overlapping) using SAT."""
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True
        
        return False
    
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner hexagons {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, f"inner vertex outside outer hexagon"
    
    return True, ""


def _min_containing_hex_side(points, outer_angle=0.0):
    """
    Compute minimum side length s of a regular hexagon (centered at origin,
    with given angle) that contains all given points.
    """
    normal_angles = np.deg2rad(outer_angle + 30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Project all points onto each normal
    projections = points @ normals.T  # shape (n_points, 6)
    
    # For containment, we need max |projection| in each normal direction
    max_abs_proj = np.max(np.abs(projections), axis=0)  # shape (6,)
    s_min = np.max(max_abs_proj) / (np.sqrt(3.0) / 2.0)
    
    return s_min


def _lattice_to_cartesian(lattice_coords, rotation_angle):
    """
    Convert hexagonal lattice coordinates to Cartesian.
    Axial coordinates (q, r):
    x = sqrt(3) * (q + r/2)
    y = 1.5 * r
    """
    q = lattice_coords[:, 0]
    r = lattice_coords[:, 1]
    
    x = np.sqrt(3.0) * (q + r / 2.0)
    y = 1.5 * r
    
    # Apply rotation
    theta = np.deg2rad(rotation_angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    
    return np.column_stack([x_rot, y_rot])


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    # Define candidate rigid clusters of 12 hexagons in axial coordinates.
    # The most compact clusters are preferred.
    
    # Cluster 1: Center + Ring 1 (6) + 5 from Ring 2 (arc)
    # This is the same as the first config in the original code.
    cluster1 = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1]
    ])
    
    # Cluster 2: Center + Ring 1 (6) + 5 from Ring 2 (different arc)
    cluster2 = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [2, -1], [1, -2], [0, -2], [-1, -1]
    ])
    
    # Cluster 3: Center + Ring 1 (6) + 5 from Ring 2 (symmetric spread)
    cluster3 = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [0, 2], [-2, 2], [-2, 0], [0, -2]
    ])
    
    # Cluster 4: A "bar" or elongated shape might fit better in a specific rotation?
    # Let's stick to compact ones first.
    
    clusters = [cluster1, cluster2, cluster3]
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    for ci, cluster in enumerate(clusters):
        if time.time() > deadline:
            break
            
        # Pre-compute base Cartesian coordinates (unrotated)
        base_cart = _lattice_to_cartesian(cluster, 0.0)
        
        def objective(params):
            theta, tx, ty = params
            # Rotate base_cart
            theta_rad = np.deg2rad(theta)
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)
            
            x = base_cart[:, 0] * cos_t - base_cart[:, 1] * sin_t
            y = base_cart[:, 0] * sin_t + base_cart[:, 1] * cos_t
            
            # Translate
            x += tx
            y += ty
            
            points = np.column_stack([x, y])
            
            # Compute min containing hex side
            s = _min_containing_hex_side(points)
            return s
        
        # Multi-start optimization
        # We can use differential_evolution for global search, then minimize for local refinement.
        # Bounds: theta in [0, 360), tx, ty in [-10, 10]
        bounds = [(0, 360), (-10, 10), (-10, 10)]
        
        try:
            # Use differential evolution for global search
            # This might be slow, so limit maxiter
            res = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-6, polish=False)
            
            if res.fun < best_s:
                best_s_candidate = res.fun
                best_params = res.x
                
                # Local refinement
                res_local = minimize(objective, best_params, method='Nelder-Mead', 
                                     options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
                
                if res_local.fun < best_s_candidate:
                    best_s_candidate = res_local.fun
                    best_params = res_local.x
                
                if best_s_candidate < best_s:
                    best_s = best_s_candidate
                    best_params = best_params
                    
                    # Construct inner array
                    theta, tx, ty = best_params
                    cart = _lattice_to_cartesian(cluster, theta)
                    cart_final = cart + np.array([tx, ty])
                    
                    inner = np.column_stack([
                        cart_final[:, 0],
                        cart_final[:, 1],
                        np.full(12, theta)
                    ])
                    outer = np.array([0.0, 0.0, 0.0])
                    
                    # Add small margin to s to ensure validity
                    s_val = best_s + 1e-5
                    valid, msg = _check_validity(inner, outer, s_val)
                    if valid:
                        best_inner = inner
                        best_outer = outer
                        best_s = s_val
                        print(f"Cluster {ci}: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
                        
        except Exception as e:
            print(f"Error in cluster {ci}: {e}")
            
    # If no valid solution found via optimization, use fallback
    if best_inner is None:
        # Use the original fallback
        inner = np.array([
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [-2.5, 0.0, 0.0],
            [1.25, 2.1650635, 0.0],
            [-1.25, 2.1650635, 0.0],
            [1.25, -2.1650635, 0.0],
            [-1.25, -2.1650635, 0.0],
            [3.75, 2.1650635, 0.0],
            [-3.75, 2.1650635, 0.0],
            [3.75, -2.1650635, 0.0],
            [-3.75, -2.1650635, 0.0],
            [0.0, -4.330127, 0.0],
        ])
        outer = np.array([0.0, 0.0, 0.0])
        s = 20.0
        return inner, outer, s

    return best_inner, best_outer, best_s
