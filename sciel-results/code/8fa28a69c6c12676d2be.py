import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    timeout = 590  # Safety margin for 600s limit
    
    # Constants
    n_hex = 11
    sqrt3 = np.sqrt(3)
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, s, angle_rad):
        angles = angle_rad + np.arange(6) * (np.pi / 3)
        x = cx + s * np.cos(angles)
        y = cy + s * np.sin(angles)
        return np.column_stack((x, y))
    
    # Helper: Check if two hexagons overlap
    # Returns a penalty value (0 if no overlap, >0 if overlap)
    def overlap_penalty(h1, h2):
        # h1, h2 are (cx, cy, angle_rad)
        # Vertices
        v1 = get_hex_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = get_hex_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # Edge normals for hex 1
        # Edge i connects vertex i and i+1
        # Normal is perpendicular to edge
        # For a regular hexagon, edge normals are at angle + 30 deg offset?
        # Vertex angles: a + i*60
        # Edge 0 (v0 to v1) is at angle a + 30
        # Normal to edge 0 is at a + 30 + 90 = a + 120? Or a - 60?
        # Let's just compute normals from vertices.
        
        def get_normals(vertices):
            normals = []
            for i in range(6):
                p1 = vertices[i]
                p2 = vertices[(i+1)%6]
                edge = p2 - p1
                # Normal
                n = np.array([-edge[1], edge[0]])
                n = n / np.linalg.norm(n)
                normals.append(n)
            return np.array(normals)
        
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        
        all_normals = np.vstack((n1, n2))
        
        min_gap = np.inf
        for n in all_normals:
            # Project v1 and v2 onto n
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap between intervals [min1, max1] and [min2, max2]
            # If max1 < min2, gap is min2 - max1
            # If max2 < min1, gap is min1 - max2
            # Else, overlap (gap is negative)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < min_gap:
                min_gap = gap
                
        # If min_gap < 0, they overlap.
        # We want a penalty for overlap.
        # The constraint is min_gap > 1e-6.
        # So if min_gap < 1e-6, we have a violation.
        # Penalty = max(0, 1e-6 - min_gap)
        # To make it smooth for optimization, use soft penalty.
        # Actually, just returning the violation amount is good for L-BFGS if we add a large multiplier.
        
        return max(0, 1e-6 - min_gap)
    
    # Helper: Calculate minimum outer hexagon side length for a given configuration
    def calc_min_outer_side(inner_params, outer_angle_rad):
        # inner_params: (11, 3) [x, y, angle_rad]
        # outer_angle_rad: angle of outer hexagon
        
        # Get all vertices of inner hexagons
        all_verts = []
        for i in range(n_hex):
            v = get_hex_vertices(inner_params[i,0], inner_params[i,1], 1.0, inner_params[i,2])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Normals of outer hexagon
        # Vertices of outer hex are at angle outer_angle + i*60
        # Edge normals are at outer_angle + i*60 + 30?
        # Let's verify.
        # Hex vertex angles: a, a+60, ...
        # Edge 0 connects a and a+60. Midpoint angle a+30.
        # Normal to edge is perpendicular to chord.
        # Chord vector angle is a+30 + 90? No.
        # Vector from center to midpoint of edge is along the normal.
        # The angle of the normal to the edge connecting vertex i and i+1 is a + 30 + i*60.
        
        normals = []
        for k in range(6):
            angle = outer_angle_rad + np.pi/6 + k * (np.pi/3)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        normals = np.array(normals)
        
        max_proj = 0
        for n in normals:
            proj = np.abs(np.dot(all_verts, n))
            max_proj = max(max_proj, np.max(proj))
            
        # Distance from center to edge is S * sqrt(3)/2
        # So S * sqrt(3)/2 >= max_proj
        # S >= 2 * max_proj / sqrt(3)
        
        return 2 * max_proj / sqrt3
        
    # Objective function for optimization
    def objective(params):
        # params: flattened array of (11 * 3 + 1)
        # First 33: inner hexagons (x, y, angle)
        # Last 1: outer hexagon angle
        
        inner_params = params[:33].reshape((11, 3))
        outer_angle = params[33]
        
        # 1. Calculate overlap penalties
        overlap_pen = 0.0
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                overlap_pen += overlap_penalty(inner_params[i], inner_params[j])
        
        # 2. Calculate outer side length
        side = calc_min_outer_side(inner_params, outer_angle)
        
        # We want to minimize side, but penalize overlaps heavily.
        # If there is overlap, the packing is invalid.
        # We use a large penalty weight.
        weight = 1000.0
        return side + weight * overlap_pen
        
    # Initialize
    # Start with a hexagonal grid
    # (0,0)
    # (2,0), (-2,0)
    # (1, sqrt(3)), (-1, sqrt(3))
    # (1, -sqrt(3)), (-1, -sqrt(3))
    # (3, sqrt(3)), (-3, sqrt(3))
    # (3, -sqrt(3)), (-3, -sqrt(3))
    
    init_positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [1.0, sqrt3, 0.0],
        [-1.0, sqrt3, 0.0],
        [1.0, -sqrt3, 0.0],
        [-1.0, -sqrt3, 0.0],
        [3.0, sqrt3, 0.0],
        [-3.0, sqrt3, 0.0],
        [3.0, -sqrt3, 0.0],
        [-3.0, -sqrt3, 0.0]
    ])
    
    init_outer_angle = 0.0
    
    init_params = np.concatenate([init_positions.flatten(), [init_outer_angle]])
    
    best_params = init_params
    best_score = objective(init_params)
    
    # Optimization
    # Use L-BFGS-B or Nelder-Mead
    # Nelder-Mead is robust for small dimensions
    # Dimension is 34.
    
    options = {
        'maxiter': 1000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    # Try a few random restarts if time permits
    n_restarts = 5
    results = []
    
    for r in range(n_restarts):
        if time.time() - start_time > timeout:
            break
            
        if r == 0:
            p0 = init_params
        else:
            # Add noise
            noise_scale = 0.5 if r == 1 else 0.1
            p0 = init_params + np.random.normal(0, noise_scale, size=34)
            # Keep outer angle in [0, 60)
            p0[33] = np.random.uniform(0, np.pi/3)
            
        try:
            res = minimize(objective, p0, method='Nelder-Mead', options=options)
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Finalize result
    inner_params = best_params[:33].reshape((11, 3))
    outer_angle = best_params[33]
    
    # Calculate final side length
    final_side = calc_min_outer_side(inner_params, outer_angle)
    
    # Check validity
    # Ensure no overlaps
    valid = True
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            if overlap_penalty(inner_params[i], inner_params[j]) > 0:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to seed-like solution if optimization failed
        # This shouldn't happen with good initialization, but just in case
        inner_params = init_positions
        outer_angle = 0.0
        final_side = calc_min_outer_side(inner_params, outer_angle)
        
    # Prepare output
    # inner: (11, 3) [x, y, angle_deg]
    inner_out = inner_params.copy()
    inner_out[:, 2] = np.degrees(inner_out[:, 2])
    
    # outer: (3,) [x, y, angle_deg]
    outer_out = np.array([0.0, 0.0, np.degrees(outer_angle)])
    
    return inner_out, outer_out, final_side

# EVOLVE-BLOCK-END
