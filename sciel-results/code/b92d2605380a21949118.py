import numpy as np
import math
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    # Initial configuration: 3x3 block + 2 extra
    # Axial to Cartesian
    def axial_to_cartesian(i, j):
        x = math.sqrt(3) * (i + j / 2.0)
        y = 1.5 * j
        return [x, y]

    # Start with a 3x3 grid centered at 0, plus (2,0) and (-2,0)
    init_axial = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            init_axial.append((i, j))
    init_axial.append((2, 0))
    init_axial.append((-2, 0))
    
    init_pos = np.array([axial_to_cartesian(i, j) for i, j in init_axial])
    
    # Center the initial positions
    center = np.mean(init_pos, axis=0)
    init_pos = init_pos - center
    
    # Flatten for optimizer
    x0 = init_pos.flatten()
    
    min_dist = math.sqrt(3)
    min_dist_sq = 3.0
    penalty_weight = 100.0
    
    def objective(params):
        # Reshape params to (N, 2)
        pts = params.reshape(-1, 2)
        
        # Check overlaps
        # Compute pairwise squared distances
        # Use broadcasting for speed
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Mask diagonal
        np.fill_diagonal(dist_sq, np.inf)
        min_d_sq = np.min(dist_sq)
        
        overlap_penalty = 0.0
        if min_d_sq < min_dist_sq:
            # Penalize based on how much they overlap
            # Linear penalty for distance violation
            violation = min_dist_sq - min_d_sq
            overlap_penalty = penalty_weight * violation
            
        # Calculate enclosing hexagon side length
        # Normals for regular hexagon (flat-top orientation)
        # Angles: 30, 90, 150, 210, 270, 330
        angles_deg = [30, 90, 150, 210, 270, 330]
        normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles_deg])
        
        # Projections
        projections = pts @ normals.T
        # For a centered hexagon, we need max(abs(projection)) for each normal?
        # No, the hexagon is centered at the centroid of the points?
        # The evaluator allows the outer hexagon to be placed anywhere.
        # To minimize the side length, the outer hexagon should be centered at the centroid of the inner points?
        # Not necessarily, but for a regular hexagon, the optimal placement for a given set of points
        # is such that the center is the centroid?
        # Actually, the minimum enclosing hexagon of a set of points is not necessarily centered at the centroid.
        # However, for a symmetric shape like a hexagon, and a roughly symmetric set of points,
        # centering at the centroid is a good approximation.
        # Let's assume the outer hexagon is centered at the centroid of the inner points.
        
        # Center points for projection calculation
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        proj = pts_centered @ normals.T
        # The distance to the side in direction n is max(proj[:, k]).
        # We need this to be <= S * sqrt(3) / 2.
        # So S >= 2 / sqrt(3) * max(proj[:, k]).
        # We take the max over all k.
        max_projs = np.max(proj, axis=0)
        # Also need to check the negative side?
        # Since normals cover all 6 directions, max(proj) in one direction is min(-proj) in the opposite?
        # No, max(proj[:, k]) is the extent in direction n_k.
        # The extent in direction -n_k is max(-proj[:, k]) = -min(proj[:, k]).
        # The required apothem is max(max(proj[:, k]), -min(proj[:, k])) for each k?
        # Actually, the hexagon is defined by |x . n_k| <= d.
        # So d >= max_k ( max_i |p_i . n_k| ).
        # So we need max over k of ( max over i of |p_i . n_k| ).
        
        abs_proj = np.abs(proj)
        max_abs_proj_per_normal = np.max(abs_proj, axis=0)
        global_max_proj = np.max(max_abs_proj_per_normal)
        
        side = 2.0 * global_max_proj / math.sqrt(3)
        
        return side + overlap_penalty

    # Run optimization
    best_result = None
    best_val = float('inf')
    
    # Multiple random restarts? Or just one from the lattice?
    # Let's try one from lattice, and maybe one slightly perturbed.
    
    # Restart 1: Lattice
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if time.time() > deadline:
        # Fallback to initial
        best_pos = init_pos
        best_val = objective(x0)
    else:
        if res.fun < best_val:
            best_val = res.fun
            best_pos = res.x.reshape(-1, 2)
            
    # Restart 2: Perturbed Lattice
    if time.time() < deadline - 20:
        x0_pert = x0 + np.random.normal(0, 0.1, x0.shape)
        res2 = minimize(objective, x0_pert, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res2.fun < best_val:
            best_val = res2.fun
            best_pos = res2.x.reshape(-1, 2)
            
    # Finalize
    # Center the final positions
    final_center = np.mean(best_pos, axis=0)
    final_pos = best_pos - final_center
    
    # Calculate final side length (without penalty)
    normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in [30, 90, 150, 210, 270, 330]])
    proj = final_pos @ normals.T
    max_abs_proj = np.max(np.abs(proj), axis=0)
    global_max_proj = np.max(max_abs_proj)
    final_side = 2.0 * global_max_proj / math.sqrt(3)
    
    # Add a small buffer to ensure no overlaps due to floating point in evaluator
    # Scale positions by 1.001
    scale = 1.001
    final_pos_scaled = final_pos * scale
    final_side_scaled = final_side * scale
    
    # Construct inner list
    inner = []
    for i in range(N):
        x, y = final_pos_scaled[i]
        # Angle is 0 (aligned with lattice)
        inner.append([x, y, 0.0])
        
    # Outer hexagon angle is 0
    # Return center, side, angle
    # The center should be (0,0) if we centered the points?
    # The evaluator places the outer hexagon at `center`.
    # If we return center=(0,0), the outer hexagon is at (0,0).
    # The inner points are centered at (0,0).
    # This is consistent.
    
    return inner, (0.0, 0.0), final_side_scaled, 0.0
# EVOLVE-BLOCK-END
