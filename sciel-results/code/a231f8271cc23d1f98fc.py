# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(vertices):
    """Return outward-pointing unit normals for each edge of a convex polygon (vertices in CCW order)."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # For CCW polygon, outward normal is (edge_y, -edge_x) normalized
        norm = np.array([edge[1], -edge[0]])
        length = np.linalg.norm(norm)
        if length > 1e-12:
            normals[i] = norm / length
    return normals

def _separation_check(inner_vertices, outer_vertices):
    """Check if inner hexagon is fully inside outer hexagon using separating axis test."""
    # Get normals for both hexagons
    inner_normals = _edge_normals(inner_vertices)
    outer_normals = _edge_normals(outer_vertices)
    
    all_normals = np.vstack([inner_normals, outer_normals])
    
    for normal in all_normals:
        inner_proj = inner_vertices @ normal
        outer_proj = outer_vertices @ normal
        
        inner_min, inner_max = np.min(inner_proj), np.max(inner_proj)
        outer_min, outer_max = np.min(outer_proj), np.max(outer_proj)
        
        # For inner to be inside outer, inner projection must be within outer projection
        if inner_min < outer_min - 1e-6 or inner_max > outer_max + 1e-6:
            return False
    
    return True

def _check_all_inside(inner_hexes, outer_hex):
    """Check if all inner hexagons are inside the outer hexagon."""
    outer_vertices = _hex_vertices(outer_hex[0], outer_hex[1], outer_hex[2], outer_hex[3])
    
    for ih in inner_hexes:
        inner_vertices = _hex_vertices(ih[0], ih[1], 1.0, ih[2])
        if not _separation_check(inner_vertices, outer_vertices):
            return False
    return True

def _check_no_overlap(inner_hexes):
    """Check that no two inner hexagons overlap using separating axis test."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            vi = _hex_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
            vj = _hex_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2])
            
            ni = _edge_normals(vi)
            nj = _edge_normals(vj)
            all_normals = np.vstack([ni, nj])
            
            overlap = True
            for normal in all_normals:
                proj_i = vi @ normal
                proj_j = vj @ normal
                
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                # If there's a gap, they don't overlap
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False
    return True

def _compute_min_outer_side(inner_hexes):
    """Compute the minimum side length of an outer hexagon that contains all inner hexagons."""
    # We need to find the minimum s such that there exists a regular hexagon of side s
    # containing all inner hexagons. This is a complex optimization problem.
    # For now, use a simple approach: compute bounding box and estimate.
    
    all_vertices = []
    for ih in inner_hexes:
        verts = _hex_vertices(ih[0], ih[1], 1.0, ih[2])
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # Compute convex hull to find extent
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(all_vertices)
        hull_points = all_vertices[hull.vertices]
    except:
        hull_points = all_vertices
    
    # The minimum enclosing regular hexagon is complex. Use a heuristic:
    # Project onto multiple angles and find the max extent.
    best_s = 0
    for angle_deg in np.arange(0, 60, 5):
        a = np.radians(angle_deg)
        # For a regular hexagon of side s, the "width" in direction theta is:
        # w(theta) = s * (cos(theta) + cos(60 - theta)) for theta in [0, 60]
        # Actually, the support function of a regular hexagon with side s centered at origin:
        # h(theta) = s * (cos(min(theta mod 60, 60 - theta mod 60)) / cos(30))
        
        # Simpler: just compute the max distance from centroid in all directions
        centroid = np.mean(hull_points, axis=0)
        vectors = hull_points - centroid
        distances = np.linalg.norm(vectors, axis=1)
        max_dist = np.max(distances)
        
        # For a regular hexagon, the apothem (inradius) is s * sqrt(3)/2
        # The circumradius is s
        # So if max_dist <= s, the hexagon of side s contains all points
        # But we need the hexagon to be aligned optimally.
        
        # The maximum distance from center to a vertex of a regular hexagon with side s is s.
        # The minimum distance (apothem) is s * sqrt(3)/2.
        # So we need s >= max_dist for the circumradius to cover all points.
        # But the hexagon shape matters. Let's be conservative.
        
        s_candidate = max_dist / (np.sqrt(3) / 2)  # This assumes the point is in the apothem direction
        best_s = max(best_s, s_candidate)
    
    return best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial symmetric arrangement: 1 center + 12 around in 2 rings
    # Ring 1: 6 hexagons at distance 2*sqrt(3) from center (touching center)
    # Ring 2: 6 hexagons at distance 4*sqrt(3) from center
    # Actually, for hexagonal packing, the distance between centers of touching hexagons is 2*sqrt(3)
    
    d1 = 2 * np.sqrt(3)  # distance for first ring
    d2 = 4 * np.sqrt(3)  # distance for second ring
    
    # Center hexagon
    centers = [[0, 0]]
    
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.radians(60 * k)
        centers.append([d1 * np.cos(angle), d1 * np.sin(angle)])
    
    # Second ring: 6 hexagons (at 30, 90, 150, 210, 270, 330 degrees)
    for k in range(6):
        angle = np.radians(60 * k + 30)
        centers.append([d2 * np.cos(angle), d2 * np.sin(angle)])
    
    # All hexagons have angle 0 initially
    inner_hexes = np.array(centers + [[0]] * 13)
    
    # Optimize using Nelder-Mead
    def objective(params):
        # params: 12*2 for centers + 12 for angles + 3 for outer (x, y, angle)
        # But we can fix outer center at origin for symmetry
        # params: 12*2 for centers + 12 for angles + 1 for outer angle + 1 for outer side
        # Actually, let's optimize: 12*2 centers, 12 angles, outer angle, outer side
        # Total: 24 + 12 + 1 + 1 = 38 parameters
        
        if time.time() > deadline - 10:
            return 1e10
        
        cx = params[0:24].reshape(12, 2)
        angles = params[24:36]
        outer_angle = params[36]
        outer_side = params[37]
        
        # Check for non-finite
        if not np.all(np.isfinite(params)):
            return 1e10
        
        # Build inner hexagons
        ih = np.column_stack([cx, angles])
        
        # Check no overlap
        if not _check_no_overlap(ih):
            return 1e10 + outer_side
        
        # Check all inside outer
        outer_hex = np.array([0, 0, outer_side, outer_angle])
        if not _check_all_inside(ih, outer_hex):
            return 1e10 + outer_side
        
        return outer_side
    
    # Initial guess
    init_params = np.concatenate([
        inner_hexes[:, 0:2].flatten(),
        inner_hexes[:, 2].flatten(),
        [0],  # outer angle
        [10]  # outer side initial guess
    ])
    
    # Use Nelder-Mead
    try:
        result = minimize(
            objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
    except Exception as e:
        # Fallback to initial guess
        result = None
    
    if result is not None and result.fun < 1e9:
        best_params = result.x
    else:
        best_params = init_params
    
    # Extract solution
    cx = best_params[0:24].reshape(12, 2)
    angles = best_params[24:36]
    outer_angle = best_params[36]
    outer_side = best_params[37]
    
    inner_hex_data = np.column_stack([cx, angles])
    outer_hex_data = np.array([0, 0, outer_angle])
    
    # Verify and adjust if needed
    if not np.isfinite(outer_side) or outer_side < np.sqrt(12):
        # Fallback
        outer_side = 10
        outer_angle = 0
        outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side

# EVOLVE-BLOCK-END
