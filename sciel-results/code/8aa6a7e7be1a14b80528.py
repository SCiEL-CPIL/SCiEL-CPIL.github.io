# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
        # Outward normal: rotate edge by -90 degrees (clockwise) and normalize
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
        # If there's a gap, they are separated
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

def _compute_s(inner_positions, outer_center, outer_angle_deg):
    """Compute the minimum side length s of outer hexagon containing all inner hexagons."""
    outer_center = np.array(outer_center)
    inner_centers = np.array(inner_positions)
    shifted = inner_centers - outer_center
    
    a = np.radians(outer_angle_deg)
    normal_angles = a + np.radians(60) * np.arange(6) + np.radians(30)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    all_vertices = []
    for i in range(12):
        cx, cy = shifted[i]
        verts = _hex_vertices(cx, cy, 1, 0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    projections = all_vertices @ normals.T
    max_proj = projections.max()
    
    s = 2 * max_proj / np.sqrt(3)
    
    return s

def _objective_with_penalties(params):
    """Objective function with smooth penalties for overlap."""
    pos = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    outer_angle = params[26]
    
    s = _compute_s(pos, outer_center, outer_angle)
    
    penalty = 0
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(pos[i,0], pos[i,1], 1, 0)
            v2 = _hex_vertices(pos[j,0], pos[j,1], 1, 0)
            n1 = _hex_edge_normals(v1)
            n2 = _hex_edge_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            min_separation = np.inf
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
                
                min_separation = min(min_separation, sep)
            
            if min_separation < 0:
                penalty += 100 * min_separation * min_separation
    
    return s + penalty

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    
    # Initial guess: Lattice-based
    # Honeycomb lattice of side 1 hexagons.
    # Distance between centers of touching hexagons is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3).
    
    # Let's generate 12 points on a triangular lattice.
    # 1, 2, 3, 4?
    # 1 in center.
    # 6 around it.
    # 5 more to make 12?
    # The next ring has 12. We take 5 of them?
    # Or a different shape.
    # A 3x4 rectangle?
    # Let's use the grid from the seed program as a starting point, but maybe optimize it.
    # Or generate a better grid.
    
    # Let's try a hexagonal cluster of 19 (1+6+12) and remove 7?
    # Or just use the seed's grid and optimize.
    
    # Seed grid:
    # (0,0)
    # (-2.5, 0), (2.5, 0)  -> dist 2.5 > sqrt(3) ~ 1.732. OK.
    # (-1.25, 2.17), (1.25, 2.17) -> dist to center: sqrt(1.5625 + 4.7) ~ 2.5. OK.
    # This is a valid packing.
    
    # Let's define the initial positions.
    # We will optimize:
    # 1. Positions of 12 hexagons (24 vars)
    # 2. Outer center (2 vars)
    # 3. Outer angle (1 var)
    # Total 27 vars.
    
    # Initial values from seed
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
    init_outer_center = np.array([0, 0])
    init_outer_angle = 0
    
    init_params = np.concatenate([init_pos.flatten(), init_outer_center, [init_outer_angle]])
    
    best_params = init_params.copy()
    best_score = _objective_with_penalties(best_params)
    
    # Optimization
    # Use L-BFGS-B or Nelder-Mead?
    # Nelder-Mead is robust for non-smooth objectives (penalty is not smooth at 0).
    # But L-BFGS-B is faster if we smooth the penalty.
    # My penalty is quadratic in penetration depth, which is continuous but not differentiable at 0?
    # Actually, the minimum of a set of linear functions is concave, so -min is convex?
    # The distance function between two convex sets is not differentiable everywhere.
    # Let's use Nelder-Mead for robustness.
    
    # To speed up, we can fix the outer center to the centroid of inner positions?
    # No, the optimal outer hexagon might not be centered at the centroid.
    # But it's a good heuristic.
    # Let's try optimizing outer center and angle separately for a given set of positions?
    # That reduces the problem to 24 variables for positions.
    # For a given set of positions, the optimal outer hexagon (min s) can be found by:
    # 1. Fix outer center to the "center of mass" of the bounding box?
    # 2. Optimize outer angle?
    
    # Let's try a nested approach:
    # 1. Optimize positions (24 vars) using Nelder-Mead, with outer hexagon fitted optimally for each step?
    # Fitting outer hexagon:
    # Given positions, find outer_center and outer_angle to minimize s.
    # s = 2/sqrt(3) * max_i max_v (n_i(angle) . (v - center)).
    # This is a minimax problem.
    # We can optimize outer_center and outer_angle (3 vars) for fixed positions using Nelder-Mead.
    
    def _fit_outer_hex(pos):
        # Optimize outer_center and outer_angle
        def obj_outer(params_outer):
            oc = params_outer[:2]
            oa = params_outer[2]
            return _compute_s(pos, oc, oa)
        
        # Initial guess for outer
        oc_init = np.mean(pos, axis=0)
        oa_init = 0
        res_outer = minimize(obj_outer, np.concatenate([oc_init, [oa_init]]), method='Nelder-Mead', 
                             options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200})
        return res_outer.fun, res_outer.x
    
    def _obj_positions(params_pos):
        pos = params_pos.reshape(12, 2)
        # Check separation penalty
        penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                v1 = _hex_vertices(pos[i,0], pos[i,1], 1, 0)
                v2 = _hex_vertices(pos[j,0], pos[j,1], 1, 0)
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
                if min_sep < 0:
                    penalty += 100 * min_sep * min_sep
        
        if penalty > 10: # Too much overlap, return large value
            return 1000 + penalty
        
        # Fit outer hex
        s, outer_params = _fit_outer_hex(pos)
        return s + penalty
    
    # Optimize positions
    # This is 24 variables. Nelder-Mead might be slow.
    # Let's try it.
    
    # To save time, we can limit the number of iterations.
    # And check the deadline.
    
    # Let's run Nelder-Mead on positions.
    # We need to be careful with the time.
    
    # Alternative: Use the seed solution and just optimize the outer hexagon?
    # The seed solution has s=8.
    # Can we fit a smaller outer hexagon to the seed positions?
    
    s_seed, outer_params_seed = _fit_outer_hex(init_pos)
    # If this is better, update best.
    if s_seed < best_score:
        best_score = s_seed
        best_params = np.concatenate([init_pos.flatten(), outer_params_seed])
    
    # Now try to optimize positions.
    # We will do a few restarts with different initializations.
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Perturb initial positions
        if r == 0:
            init_p = init_pos
        else:
            init_p = init_pos + np.random.normal(0, 0.5, init_pos.shape)
        
        # Run Nelder-Mead
        # We need to wrap the objective to check time?
        # Nelder-Mead doesn't support callback for time check easily.
        # We can just run it and hope it's fast enough.
        # Or limit maxiter.
        
        try:
            res = minimize(_obj_positions, init_p.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500, 'maxfev': 2000})
            
            if res.fun < best_score:
                best_score = res.fun
                best_pos = res.x.reshape(12, 2)
                # Refit outer hex for best positions
                s_best, outer_params_best = _fit_outer_hex(best_pos)
                best_params = np.concatenate([best_pos.flatten(), outer_params_best])
                best_score = s_best # Update with the actual s without penalty
                
        except Exception as e:
            pass # Ignore errors in optimization
        
        # Check deadline
        if time.time() > deadline - 60:
            break
    
    # Final check: ensure validity
    # Extract best solution
    best_pos = best_params[:24].reshape(12, 2)
    best_outer_center = best_params[24:26]
    best_outer_angle = best_params[26]
    
    # Calculate final s
    final_s = _compute_s(best_pos, best_outer_center, best_outer_angle)
    
    # Check if inner hexes are inside
    # And if they are separated.
    # If not, fall back to seed?
    # The seed is known to be valid (score 1.0).
    # Let's check validity of the optimized solution.
    
    valid = True
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(best_pos[i,0], best_pos[i,1], 1, 0)
            v2 = _hex_vertices(best_pos[j,0], best_pos[j,1], 1, 0)
            if not _separation_check(v1, v2):
                valid = False
                break
        if not valid:
            break
    
    # Check inside outer
    if valid:
        outer_verts = _hex_vertices(best_outer_center[0], best_outer_center[1], final_s, best_outer_angle)
        for i in range(12):
            inner_verts = _hex_vertices(best_pos[i,0], best_pos[i,1], 1, 0)
            if not _is_inside_outer(inner_verts, outer_verts):
                valid = False
                break
    
    if not valid:
        # Fall back to seed
        best_pos = init_pos
        best_outer_center = np.array([0, 0])
        best_outer_angle = 0
        final_s = 8 # Seed s
    
    # Construct return values
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_pos[i, 0]
        inner_hex_data[i, 1] = best_pos[i, 1]
        inner_hex_data[i, 2] = 0 # Angle
    
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    return inner_hex_data, outer_hex_data, final_s

# EVOLVE-BLOCK-END
