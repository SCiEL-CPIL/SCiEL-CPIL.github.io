# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_hex_vertices(cx, cy, side, angle_deg):
    """
    Returns vertices of a regular hexagon.
    """
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _min_distance_polygons(p1, p2):
    """
    Compute the minimum distance between two convex polygons p1 and p2.
    If they intersect, returns a negative value (penetration depth).
    Uses Separating Axis Theorem (SAT).
    """
    # Get edges normals for p1 and p2
    def get_normals(p):
        normals = []
        for i in range(len(p)):
            p1 = p[i]
            p2 = p[(i + 1) % len(p)]
            edge = p2 - p1
            # Normal perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            normals.append(n)
        return normals

    n1 = get_normals(p1)
    n2 = get_normals(p2)
    
    min_penetration = float('inf')
    
    for n in n1 + n2:
        # Project p1 and p2 onto n
        proj1 = np.dot(p1, n)
        proj2 = np.dot(p2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for overlap
        if max1 < min2 or max2 < min1:
            # Separated
            return float('inf') # Or a large positive number indicating no overlap
        else:
            # Overlap on this axis
            # Penetration depth is the amount of overlap
            # If intervals [min1, max1] and [min2, max2] overlap:
            # Intersection is [max(min1, min2), min(max1, max2)]
            # The "distance" is 0 if they overlap? 
            # For SAT, if they overlap on all axes, they intersect.
            # We want a measure of "how much" they overlap to use as a penalty.
            # A common metric is the minimum overlap depth across all axes.
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_penetration:
                min_penetration = overlap
                
    # If we found an axis where they are separated, min_penetration would be inf?
    # No, the loop returns inf immediately if separated.
    # If we finish the loop, they intersect.
    # The minimum penetration depth is min_penetration.
    # We want to minimize this (make it <= 0).
    # So the violation is max(0, min_penetration).
    return min_penetration

def _point_in_polygon(p, poly):
    """
    Check if point p is inside polygon poly.
    Returns signed distance (positive if inside, negative if outside).
    For a regular hexagon, we can use the support function.
    """
    # This is a generic check, but we know poly is a regular hexagon.
    # We'll use the edge normals method.
    # If p is to the "outside" of any edge, it's outside.
    # Signed distance to edge = dot(p - p_edge_start, normal)
    
    min_dist = float('inf')
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i + 1) % len(poly)]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        n = n / np.linalg.norm(n)
        
        # Ensure normal points inward
        # Check centroid
        centroid = np.mean(poly, axis=0)
        if np.dot(n, centroid - p1) < 0:
            n = -n
            
        dist = np.dot(p - p1, n)
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    n_inner = 12
    inner_side = 1.0
    
    # Initial guess: Compact cluster
    # Centers in a hexagonal lattice
    # Basis vectors for touching hexagons (side 1)
    # Distance = sqrt(3)
    d = np.sqrt(3)
    u = np.array([d, 0])
    v = np.array([d/2, d*np.sqrt(3)/2])
    
    # Generate 12 positions
    # 1 center, 6 neighbors, 5 more
    centers = []
    centers.append(np.array([0.0, 0.0]))
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        # Rotate u and v?
        # Simpler: use grid coords
        pass
    
    # Let's use explicit grid coords (q, r)
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2 * r)  -- Wait, standard hex grid formula
    # For pointy top:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # Here size=1.
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        # Add 5 more from ring 2
        (2, 0), (2, -1), (1, -2), (-1, -2), (-2, -1)
    ]
    
    init_centers = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        init_centers.append([x, y])
        
    init_centers = np.array(init_centers)
    
    # Normalize init_centers to be centered at origin
    init_centers -= np.mean(init_centers, axis=0)
    
    # Initial outer hexagon
    # Estimate bounding box
    max_x = np.max(np.abs(init_centers[:, 0]))
    max_y = np.max(np.abs(init_centers[:, 1]))
    # Approximate s
    init_s = max(max_x, max_y) * 1.5 + 2.0
    
    init_outer_center = np.array([0.0, 0.0])
    init_outer_angle = 0.0
    
    # Flatten parameters
    # 12 centers (24), outer center (2), outer angle (1), s (1)
    x0 = np.concatenate([init_centers.flatten(), init_outer_center, [init_outer_angle], [init_s]])
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
            
        centers = params[0:24].reshape(12, 2)
        outer_center = params[24:26]
        outer_angle = params[26]
        s = params[27]
        
        # Ensure s is positive
        if s < 1.0:
            return 1e6
            
        # Calculate inner vertices
        inner_vertices = []
        for i in range(12):
            cx, cy = centers[i]
            # Angle 0 for all inner hexagons
            verts = _get_hex_vertices(cx, cy, 1.0, 0.0)
            inner_vertices.append(verts)
            
        # Calculate outer vertices
        outer_vertices = _get_hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
        
        penalty = 0.0
        
        # 1. Containment
        for i in range(12):
            for v in inner_vertices[i]:
                dist = _point_in_polygon(v, outer_vertices)
                if dist < 0:
                    penalty += dist * dist
        
        # 2. Separation
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick check: distance between centers
                dc = np.linalg.norm(centers[i] - centers[j])
                if dc < 2 * np.sqrt(3): # Approx bound
                    dist = _min_distance_polygons(inner_vertices[i], inner_vertices[j])
                    if dist < 0:
                        penalty += dist * dist
        
        return s + 10.0 * penalty

    # Optimization
    best_params = x0
    best_val = objective(x0)
    
    # Use Nelder-Mead
    options = {'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass
        
    # Try another start point if time permits
    if time.time() < deadline - 100:
        # Perturb initial centers
        np.random.seed(42)
        x1 = x0.copy()
        x1[0:24] += np.random.normal(0, 0.1, 24)
        x1[27] = init_s * 0.9 # Try smaller s
        
        best_val_2 = objective(x1)
        if best_val_2 < best_val:
            best_params = x1
            best_val = best_val_2
            
        try:
            res2 = minimize(objective, x1, method='Nelder-Mead', options=options)
            if res2.fun < best_val:
                best_params = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Extract best result
    centers = best_params[0:24].reshape(12, 2)
    outer_center = best_params[24:26]
    outer_angle = best_params[26]
    s = best_params[27]
    
    # Final Validation and Cleanup
    # Ensure s is at least sqrt(12)
    s = max(s, np.sqrt(12))
    
    # Construct return arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0 # Angle
        
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
