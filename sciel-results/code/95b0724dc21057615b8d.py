import numpy as np
from scipy.optimize import minimize
import itertools

def hexagon_vertices(cx, cy, side, angle_deg):
    """
    Calculate vertices of a regular hexagon.
    Vertices are at cx + side * cos(angle + 60*i), cy + side * sin(angle + 60*i)
    """
    angles = np.deg2rad(angle_deg + np.arange(6) * 60)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def sat_gap(A, B, normals):
    """
    Calculate the maximum gap between two polygons A and B along the given normals.
    If max gap > 0, they are separated.
    If max gap < 0, they overlap.
    Returns the max gap.
    """
    # A, B are (6, 2) arrays of vertices
    # normals are (N, 2) array of unit vectors
    
    # Projection of A onto normals: min dot product
    # proj_A = np.min(A @ normals.T, axis=0)
    # Projection of B onto normals: max dot product
    # proj_B = np.max(B @ normals.T, axis=0)
    
    # Gap = min_A - max_B
    # We want the max of these gaps over all normals
    # If this max is negative, there is no separating axis, so they overlap.
    
    proj_A = np.min(A @ normals.T, axis=0)
    proj_B = np.max(B @ normals.T, axis=0)
    gaps = proj_A - proj_B
    return np.max(gaps)

def get_normals(vertices):
    """
    Get outward edge normals for a convex polygon given in CCW order.
    """
    edges = np.roll(vertices, -1, axis=0) - vertices
    # Edge vectors
    # Normal is perpendicular to edge.
    # For CCW polygon, outward normal is (dy, -dx) normalized?
    # Let's check.
    # Edge from (0,0) to (1,0). Normal (0, -1) is inward for a polygon above?
    # Standard CCW: (0,0) -> (1,0) -> (1,1).
    # Edge 1: (1,0). Outward normal should be (0, -1).
    # Edge 2: (0,1). Outward normal should be (1, 0).
    # Formula: n = (dy, -dx) for edge (dx, dy).
    # Edge (1,0) -> n=(0, -1). Correct.
    # Edge (0,1) -> n=(1, 0). Correct.
    
    dx = edges[:, 0]
    dy = edges[:, 1]
    normals = np.column_stack((dy, -dx))
    # Normalize
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]
    return normals

def calculate_penalty(params, s_outer):
    """
    params: flat array of 36 floats (x, y, angle) * 12
    s_outer: side length of outer hexagon
    """
    n_inner = 12
    penalty = 0.0
    
    # Reshape params
    inner_params = params.reshape(n_inner, 3)
    
    # Generate vertices and normals for inner hexagons
    inner_verts = []
    inner_normals = []
    for i in range(n_inner):
        x, y, ang = inner_params[i]
        v = hexagon_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
        inner_normals.append(get_normals(v))
        
    # Outer hexagon vertices and normals
    outer_v = hexagon_vertices(0, 0, s_outer, 0)
    outer_normals = get_normals(outer_v)
    
    # 1. Inner-Inner Overlaps
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            A = inner_verts[i]
            B = inner_verts[j]
            # Use normals from both A and B
            normals = np.vstack((inner_normals[i], inner_normals[j]))
            gap = sat_gap(A, B, normals)
            if gap < 0:
                # Overlap depth is -gap
                penalty += (-gap) ** 2
                
    # 2. Inner-Outer Overlaps
    # Check if any inner vertex is outside the outer hexagon
    # An outer hexagon contains a point if the point is on the "inner" side of all outer edges.
    # For an outer edge with outward normal n and point p on the edge:
    # (point - p) . n <= 0
    # Violation = max(0, (point - p) . n)
    
    for i in range(n_inner):
        v = inner_verts[i]
        # For each vertex of inner hexagon i
        for k in range(6):
            pt = v[k]
            for m in range(6):
                # Outer edge m
                p1 = outer_v[m]
                n = outer_normals[m]
                # Distance of pt to the line defined by p1 and n
                # If pt is outside, (pt - p1) . n > 0
                dist = np.dot(pt - p1, n)
                if dist > 0:
                    penalty += dist ** 2
                    
    return penalty

def hexagon_packing_12():
    n = 12
    
    # Initial guess: Honeycomb pattern
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 5 hexes? No, 12 total.
    # 1 + 6 + 5 = 12.
    # Or 1 + 6 + 1 + 4?
    # Let's just use a grid-like init scaled to a reasonable s.
    
    # Better init:
    # (0,0)
    # (1.5, 0), (-1.5, 0), (0, 2.598), (0, -2.598), (1.5, 2.598), (-1.5, 2.598)
    # This is 7 hexes.
    # Add 5 more around.
    
    init_params = np.array([
        [0, 0, 0],
        [1.5, 0, 0],
        [-1.5, 0, 0],
        [0, 2.598, 0],
        [0, -2.598, 0],
        [1.5, 2.598, 0],
        [-1.5, 2.598, 0],
        [3.0, 2.598, 0],
        [-3.0, 2.598, 0],
        [3.0, -2.598, 0],
        [-3.0, -2.598, 0],
        [0, -5.196, 0]
    ]).flatten()
    
    # Binary search for s
    s_low = np.sqrt(12) * 1.01 # Slightly above theoretical min
    s_high = 8.0
    
    best_s = s_high
    best_params = init_params
    best_penalty = calculate_penalty(best_params, best_s)
    
    # We will try to find a valid packing for various s
    # Iterate s from high to low? Or binary search.
    # Since the landscape is rugged, let's try a few fixed values first to get a good baseline, then refine.
    
    # Try to optimize for a specific s
    def try_pack(s_target, num_restarts=3):
        best_local_s = s_target
        best_local_params = None
        min_pen = float('inf')
        
        for r in range(num_restarts):
            # Jitter initial positions slightly for restarts
            if r == 0:
                p0 = init_params.copy()
            else:
                p0 = init_params + np.random.normal(0, 0.1, size=init_params.shape)
            
            # Bounds: Keep centers within reasonable range to prevent flying away
            # x, y in [-s_target, s_target], angle in [0, 360]
            bounds = []
            for i in range(n):
                bounds.append((-s_target, s_target))
                bounds.append((-s_target, s_target))
                bounds.append((0, 360))
                
            # Use Powell method (derivative free, robust for non-smooth)
            # maxiter needs to be high enough
            res = minimize(calculate_penalty, p0, args=(s_target,), method='Powell', 
                           bounds=bounds, options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < min_pen:
                min_pen = res.fun
                best_local_params = res.x
                
        if min_pen < 1e-6:
            return True, best_local_params
        else:
            return False, best_local_params

    # Check if current best is valid
    if best_penalty < 1e-6:
        valid, best_params = try_pack(best_s)
        if valid:
            pass # Already good
    
    # Binary search
    for _ in range(15): # 15 iterations of binary search
        s_mid = (s_low + s_high) / 2
        valid, params = try_pack(s_mid, num_restarts=2)
        if valid:
            best_s = s_mid
            best_params = params
            s_low = s_mid
        else:
            s_high = s_mid
            
    # Final optimization with best_s
    valid, best_params = try_pack(best_s, num_restarts=5)
    if not valid:
        # Fallback to a known good configuration if optimization failed
        # Use the seed's configuration but with a tighter outer hexagon
        # The seed had s=8. Let's try to optimize the seed's positions with s=5.
        init_params = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ]).flatten()
        valid, best_params = try_pack(5.0, num_restarts=5)
        if valid:
            best_s = 5.0
        else:
            best_s = 8.0
            best_params = init_params

    # Construct return values
    inner_hex_data = best_params.reshape(12, 3)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_s
