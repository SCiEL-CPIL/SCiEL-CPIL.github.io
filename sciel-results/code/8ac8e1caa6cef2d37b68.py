import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    Assumes inner hexagons are aligned with angle 0.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    
    # Outward normals for the 6 edges of the outer hexagon
    # Vertex i is at angle a + 60i. Edge i connects V_i and V_{i+1}.
    # The outward normal to edge i points in the direction of the angle bisector of V_i and V_{i+1}.
    # Angle of V_i is a + 60i. Angle of V_{i+1} is a + 60(i+1).
    # Bisector angle is a + 60i + 30.
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    # For each normal, find the maximum projection of all inner vertices.
    # The constraint is max_proj <= s * apothem = s * sqrt(3)/2.
    # So s >= max_proj / (sqrt(3)/2).
    
    projections = all_verts_flat @ normals.T 
    max_projections = np.max(projections, axis=0)
    
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def _check_validity(positions, angle_deg, s):
    """
    Check if the configuration is valid.
    1. All inner hexagons are separated (no overlap).
    2. All inner hexagons are inside the outer hexagon.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # 1. Check separation (SAT)
    # Precompute vertices for all inner hexagons
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :] # (12, 6, 2)
    
    # Precompute edge normals for a unit hexagon (angle 0)
    # Edges of unit hexagon:
    # V0(1,0) to V1(0.5, sqrt(3)/2) -> e=(-0.5, sqrt(3)/2). Normal (sqrt(3)/2, 0.5) or (-sqrt(3)/2, -0.5).
    # Let's just compute normals from vertices for robustness.
    edge_normals = []
    for k in range(6):
        v1 = unit_hex_verts[k]
        v2 = unit_hex_verts[(k+1)%6]
        e = v2 - v1
        n = np.array([-e[1], e[0]])
        n = n / np.linalg.norm(n)
        edge_normals.append(n)
    edge_normals = np.array(edge_normals)
    
    for i in range(12):
        for j in range(i+1, 12):
            verts_i = all_verts[i]
            verts_j = all_verts[j]
            
            # Check axes from i and j (which are same since same orientation)
            separated = False
            for n in edge_normals:
                proj_i = np.dot(verts_i, n)
                proj_j = np.dot(verts_j, n)
                
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
                
    # 2. Check containment
    # Outer hexagon vertices
    outer_verts = _hex_vertices(0, 0, s, angle_deg)
    
    # For each inner vertex, check if it's inside outer hexagon
    # Using cross product method: cross(p2 - p1, v - p1) >= -eps for all edges in CCW order
    # Note: _hex_vertices generates vertices in CCW order if angle increases.
    # cos/sin with increasing angle is CCW.
    
    for i in range(12):
        verts_i = all_verts[i]
        for v in verts_i:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                # Cross product e x (v - p1)
                # In 2D: e_x * v_y' - e_y * v_x'
                v_rel = v - p1
                cross_val = e[0] * v_rel[1] - e[1] * v_rel[0]
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    # Generate diverse initial configurations
    # We assume all inner hexagons are aligned (angle 0) to reduce dimensionality.
    # The outer hexagon angle is a variable.
    
    configs = []
    
    # 1. Hexagonal cluster (7 center + 5 surrounding)
    # Standard hex grid spacing: dx = sqrt(3), dy = 1.5
    h = np.sqrt(3) / 2
    # Centers:
    # (0,0)
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # Plus 5 more to make 12.
    # Let's try a 2x3 rectangle of hexagons?
    # Or a larger hexagon with 1 missing?
    # 1+6+12 = 19.
    # 1+6 = 7.
    # Let's just use a random scatter as a fallback, plus structured ones.
    
    # Config A: Tight hexagonal packing (approx)
    # 1 center, 6 neighbors, 5 next ring.
    pos_a = np.array([
        [0, 0],
        [np.sqrt(3), 0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5],
        # Next ring candidates
        [2*np.sqrt(3), 0],
        [3*np.sqrt(3)/2, 1.5],
        [np.sqrt(3)/2, 3],
        [-np.sqrt(3)/2, 3],
        [-3*np.sqrt(3)/2, 1.5],
        [-2*np.sqrt(3), 0]
    ])
    configs.append(pos_a)
    
    # Config B: Linear strip (2 rows of 6)
    # Spacing sqrt(3) in x, 1.5 in y? No, hexagons don't stack perfectly in straight lines without rotation or specific offsets.
    # If aligned, horizontal neighbors need dx = sqrt(3).
    # Vertical neighbors? If dx=0, dy must be large enough?
    # Actually, aligned hexagons in a grid:
    # (0,0), (sqrt(3), 0), ...
    # (sqrt(3)/2, 1.5), (3sqrt(3)/2, 1.5), ...
    # Let's make a 2x6 grid.
    pos_b = []
    for row in range(2):
        for col in range(6):
            x = col * np.sqrt(3)
            y = row * 1.5
            # Offset for hex grid
            if row % 2 == 1:
                x += np.sqrt(3)/2
            pos_b.append([x, y])
    configs.append(np.array(pos_b))
    
    # Config C: 3x4 Grid
    pos_c = []
    for row in range(3):
        for col in range(4):
            x = col * np.sqrt(3)
            y = row * 1.5
            if row % 2 == 1:
                x += np.sqrt(3)/2
            pos_c.append([x, y])
    configs.append(np.array(pos_c))
    
    # Config D: Random scatter
    np.random.seed(42)
    pos_d = np.random.uniform(-3, 3, (12, 2))
    configs.append(pos_d)
    
    # Config E: "Flower" shape with holes?
    # Just use the ones above.
    
    best_s = np.inf
    best_params = None
    
    # For each config, run Nelder-Mead
    # Variables: x1, y1, ..., x12, y12, outer_angle
    # Total 25 variables.
    
    def objective(params):
        pos = params[:24].reshape(12, 2)
        angle = params[24]
        
        # Quick check for validity to penalize heavily if invalid
        # We can't do full SAT in objective for speed? 
        # Nelder-Mead evaluates many times. 
        # If we return a penalty for overlap, it might get stuck.
        # Better: Just minimize s. If it overlaps, s might be small but invalid.
        # We must ensure the final result is valid.
        # During optimization, if we allow overlaps, the "minimum enclosing hexagon" of overlapping shapes is smaller.
        # This is a local minimum trap.
        # We need to penalize overlap.
        
        # Fast overlap check:
        # Distance between centers < sqrt(3) - epsilon implies potential overlap.
        # Actually, for aligned hexagons, if dist > 2*apothem = sqrt(3), they don't overlap?
        # No, apothem is sqrt(3)/2. Distance between centers of touching hexagons is sqrt(3).
        # If dist < sqrt(3), they overlap?
        # Let's check.
        # Two hexagons at (0,0) and (sqrt(3)-eps, 0).
        # Rightmost point of left is (1,0). Leftmost point of right is (sqrt(3)-eps - 1, 0) = (0.732-eps, 0).
        # 1 > 0.732. They overlap.
        # So if dist < sqrt(3), they overlap (for aligned hexagons along x-axis).
        # What about diagonal?
        # Distance between (0,0) and (sqrt(3)/2, 1.5) is sqrt(3/4 + 2.25) = sqrt(3).
        # They touch.
        # So for aligned hexagons, min distance is sqrt(3).
        
        # Penalty if any pair is closer than sqrt(3) - 1e-4
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
        
        if min_dist < np.sqrt(3) - 1e-4:
            # Penalty proportional to overlap
            return _min_outer_side(pos, angle) + 100 * (np.sqrt(3) - min_dist)
            
        return _min_outer_side(pos, angle)
    
    for i, init_pos in enumerate(configs):
        if time.time() > deadline:
            break
            
        init_params = np.concatenate([init_pos.flatten(), [0]])
        
        # Try multiple outer angles for each config?
        # Or let the optimizer find the angle.
        # Let's fix a few angles for initial runs to help convergence.
        angles_to_try = [0, 30, 60, 90, 120, 150]
        
        for ang in angles_to_try:
            if time.time() > deadline:
                break
                
            init_params[24] = ang
            
            try:
                res = minimize(objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6})
                
                if res.fun < best_s:
                    # Validate the result
                    pos = res.x[:24].reshape(12, 2)
                    ang = res.x[24]
                    s = _min_outer_side(pos, ang)
                    
                    if _check_validity(pos, ang, s):
                        best_s = s
                        best_params = res.x
                        
            except Exception as e:
                pass
                
    if best_params is None:
        # Fallback to a simple valid configuration
        # Use the first config and calculate s
        pos = configs[0]
        ang = 0
        s = _min_outer_side(pos, ang)
        # Ensure validity, if not, spread them out
        # For fallback, just return a large valid box
        inner_hex_data = np.zeros((12, 3))
        inner_hex_data[:, :2] = pos
        inner_hex_data[:, 2] = 0
        outer_hex_data = np.array([0, 0, ang])
        # Make s large enough to be safe
        s = max(s, 10)
        return inner_hex_data, outer_hex_data, s
        
    # Construct return values
    pos = best_params[:24].reshape(12, 2)
    ang = best_params[24]
    s = _min_outer_side(pos, ang)
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, :2] = pos
    inner_hex_data[:, 2] = 0 # Aligned
    
    outer_hex_data = np.array([0, 0, ang])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
