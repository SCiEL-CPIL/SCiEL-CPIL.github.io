import numpy as np
from scipy.optimize import minimize_scalar
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 630.0
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return x, y
    
    # Precompute hexagon vertices for a unit hexagon at origin, angle 0
    hex_angles = np.deg2rad(np.arange(0, 360, 60))
    hex_vertices_unit = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # Normal angles for outer hexagon (angle 0): normals at 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Candidate configurations of 12 cells
    # Config A: current (seed-like)
    config_A = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, 0), (-2, 0),
        (1, -2), (-1, 2)
    ]
    
    # Config B: Full ring 1 (7) + 5 from ring 2, trying to be more compact
    # Ring 1: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) = 7 cells
    # Ring 2 has 12 cells. Pick 5 that are most "inward" or compact.
    # Actually, let's try a different shape: a 4x3 rectangle in axial? No, that won't be hexagonal.
    # Let's try: center + ring 1 (6) + 5 more from ring 2 that form a connected compact shape
    # Or try: a "thick" hexagon. Side 2 hex has 19 cells. Remove 7 to get 12.
    # Which 7 to remove to minimize bounding hexagon?
    # The corners of side-2 hex are at distance 2*sqrt(3) from center.
    # Remove the 6 corners and 1 more? Or remove 7 in a symmetric way?
    
    # Let me enumerate some promising 12-cell shapes:
    # Shape 1: Side 2 hex (19) minus 6 corners minus 1 edge-mid
    # Corners of side 2 hex in axial (pointy top, but our coords are flat-top?):
    # With x=1.5q, y=sqrt(3)(r+q/2):
    # The 6 "corners" of a side-2 hexagon are at:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # Let's verify: (2,0) -> x=3, y=sqrt(3)*1 = 1.732
    # (0,2) -> x=0, y=2*sqrt(3) = 3.464
    # These are the 6 vertices of the hexagonal lattice cell of side 2.
    
    config_B = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (1, -2), (-1, 2),
        (0, 2), (0, -2)
    ]
    # This is 13 cells. Remove one. Which one? Maybe (0,2) or (0,-2) to make it 12.
    config_B = config_B[:12]  # Remove (0,-2)
    
    # Config C: Try a more linear/elongated shape that might fit better with rotation
    config_C = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, 0), (-2, 0),
        (3, -1), (-3, 1)
    ]
    
    # Config D: Dense cluster - center + ring 1 + 5 adjacent ring 2 cells
    config_D = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, -2), (-1, 2), (-2, 1), (2, 0)
    ]
    
    configs = [config_A, config_B, config_C, config_D]
    
    def calc_s_smooth(theta, points, smooth_beta=100.0):
        """Calculate s using smooth max (log-sum-exp) for differentiability."""
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        rot_centers = points @ rot.T
        rot_hex_verts = hex_vertices_unit @ rot.T
        
        # All vertices: (12*6, 2)
        all_verts = np.zeros((12 * 6, 2))
        for i in range(12):
            all_verts[i*6:(i+1)*6] = rot_centers[i] + rot_hex_verts
        
        # Project onto all 6 normals: (72, 6)
        projs = all_verts @ normals.T
        
        # Smooth max over all 72*6 = 432 projections
        # Use log-sum-exp for numerical stability
        max_projs = np.max(projs, axis=0)  # (6,)
        # For smoothness, use soft-max over the 72 vertices for each normal
        # Actually, we want max over ALL vertices and ALL normals
        # The s is determined by max over all (vertex, normal) pairs
        # Let's use a two-level approach: for each normal, soft-max over vertices, then hard max over normals
        # Or just use a large beta on the full 432 values
        
        flat_projs = projs.flatten()
        # Log-sum-exp for smooth max
        m = np.max(flat_projs)
        smooth_max = m + np.log(np.sum(np.exp(flat_projs - m))) / len(flat_projs)
        # Actually, for a true smooth max, we want:
        # log(sum(exp(projs/beta))) * beta
        # But this is expensive. Let's use a simpler approach:
        # Since the max is over a finite set, and we're just optimizing theta,
        # the non-smoothness is at measure-zero points. minimize_scalar with bounded
        # should handle it. Let's keep it simple but add a small perturbation.
        
        # Actually, let's just compute the exact max - it's fast enough
        exact_max = np.max(projs)
        s_val = exact_max * 2.0 / np.sqrt(3)
        return s_val
    
    def calc_s_exact(theta, points):
        """Exact calculation of s."""
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        rot_centers = points @ rot.T
        rot_hex_verts = hex_vertices_unit @ rot.T
        
        # For each inner hexagon, project all 6 vertices onto all 6 normals
        # We need the max over all 12*6*6 = 432 projections
        max_proj = -np.inf
        for i in range(12):
            verts = rot_centers[i] + rot_hex_verts
            projs = verts @ normals.T
            hex_max = np.max(projs)
            if hex_max > max_proj:
                max_proj = hex_max
        
        s_val = max_proj * 2.0 / np.sqrt(3)
        return s_val
    
    best_s = np.inf
    best_theta = 0.0
    best_config_idx = 0
    
    for cfg_idx, axial_coords in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Convert to Cartesian
        points = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
        
        # Scale to ensure separation: gap = (scale-1)*sqrt(3) > 1e-6
        scale = 1.0 + 1e-5
        points = points * scale
        
        # Center at origin
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Optimize theta in [0, 60 degrees) due to 6-fold symmetry of outer hex
        def objective(theta):
            return calc_s_exact(theta, points)
        
        try:
            res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', 
                                  options={'xatol': 1e-10, 'maxiter': 200})
            theta_opt = res.x
            s_val = objective(theta_opt)
            
            if s_val < best_s:
                best_s = s_val
                best_theta = theta_opt
                best_config_idx = cfg_idx
                best_points = points.copy()
                
        except Exception as e:
            continue
    
    # Final high-precision polish on the best configuration
    if time.time() < deadline - 10:
        # Use a finer grid around the optimum
        theta_coarse = best_theta
        theta_fine_start = max(0, theta_coarse - 1e-4)
        theta_fine_end = min(np.pi/3, theta_coarse + 1e-4)
        
        def objective_fine(theta):
            return calc_s_exact(theta, best_points)
        
        try:
            res_fine = minimize_scalar(objective_fine, bounds=(theta_fine_start, theta_fine_end), 
                                       method='bounded', options={'xatol': 1e-12, 'maxiter': 1000})
            theta_final = res_fine.x
            s_final = objective_fine(theta_final)
            
            if s_final < best_s:
                best_s = s_final
                best_theta = theta_final
        except:
            pass
    
    # Construct final result
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    rot = np.array([[c, -s], [s, c]])
    final_points = best_points @ rot.T
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_points[i, 0]
        inner_hex_data[i, 1] = final_points[i, 1]
        inner_hex_data[i, 2] = np.rad2deg(best_theta)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s
