import math
import itertools
import time
import numpy as np

N = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    # Hexagon geometry constants
    d = math.sqrt(3)  # distance between centers of adjacent unit hexagons
    
    # Generate centers: center + ring 1 (6) + ring 2 (12)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
    
    # Ring 2: 12 hexagons
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
    
    # Base centers (center + ring 1)
    base_centers = centers[:7]
    
    # Precompute vertices relative to center (unit hexagon, pointy-top)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    # Precompute base points (center + ring 1 vertices)
    base_points = []
    for cx, cy in base_centers:
        for vx, vy in verts_rel:
            base_points.append((cx + vx, cy + vy))
    base_points_arr = np.array(base_points)  # shape (42, 2)
    
    # Precompute ring 2 points for each of the 12 positions
    ring2_points_list = []
    for cx, cy in ring2_centers:
        pts = []
        for vx, vy in verts_rel:
            pts.append((cx + vx, cy + vy))
        ring2_points_list.append(np.array(pts))  # each shape (6, 2)
    
    # All combinations of 4 from 12 ring 2 positions
    all_combinations = list(itertools.combinations(range(12), 4))
    
    # For each combination, we need to find min over theta of max over 6 normals of max over 66 points of |dot|
    # S(theta) = (2/sqrt(3)) * max_k max_i |P_i . n_k(theta)|
    
    # Vectorized approach:
    # For a given combination, points = vstack(base_points, ring2_points[combo])
    # We sweep theta and compute the max projection
    
    best_S = float('inf')
    best_combo = None
    best_theta = 0.0
    
    # Coarse sweep: 500 steps over [0, pi/3)
    n_coarse = 500
    theta_coarse = np.linspace(0, math.pi / 3, n_coarse, endpoint=False)
    k_angles = np.arange(6) * math.pi / 3
    
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
        
        # Build points array for this combination
        combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])  # (66, 2)
        
        # Compute all dot products: for each theta, each normal, each point
        # theta_coarse: (500,)
        # k_angles: (6,)
        # combo_points: (66, 2)
        
        # angles shape: (500, 6)
        angles = theta_coarse[:, np.newaxis] + k_angles[np.newaxis, :]
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (500, 6, 2)
        
        # dots: (500, 6, 66)
        # normals: (500, 6, 1, 2), points: (1, 1, 66, 2)
        dots = np.sum(normals[:, :, np.newaxis, :] * combo_points[np.newaxis, np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        
        # max over points: (500, 6)
        max_per_normal = np.max(abs_dots, axis=-1)
        # max over normals: (500,)
        max_proj = np.max(max_per_normal, axis=-1)
        S_vals = (2.0 / math.sqrt(3)) * max_proj
        
        min_idx = np.argmin(S_vals)
        min_S = S_vals[min_idx]
        min_theta = theta_coarse[min_idx]
        
        if min_S < best_S:
            best_S = min_S
            best_combo = combo
            best_theta = min_theta
    
    # Refinement pass on the best combination
    if best_combo is not None and time.time() < deadline - safety_margin:
        # Fine sweep: 5000 steps over a window around best_theta
        step_size = (math.pi / 3) / n_coarse
        window = 3 * step_size
        
        # Fine sweep over the full range with higher resolution
        n_fine = 5000
        theta_fine = np.linspace(0, math.pi / 3, n_fine, endpoint=False)
        
        # Build points for best combo
        best_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in best_combo])
        
        angles_fine = theta_fine[:, np.newaxis] + k_angles[np.newaxis, :]
        normals_fine = np.stack([np.cos(angles_fine), np.sin(angles_fine)], axis=-1)
        
        dots_fine = np.sum(normals_fine[:, :, np.newaxis, :] * best_points[np.newaxis, np.newaxis, :, :], axis=-1)
        abs_dots_fine = np.abs(dots_fine)
        max_per_normal_fine = np.max(abs_dots_fine, axis=-1)
        max_proj_fine = np.max(max_per_normal_fine, axis=-1)
        S_fine = (2.0 / math.sqrt(3)) * max_proj_fine
        
        fine_idx = np.argmin(S_fine)
        if S_fine[fine_idx] < best_S:
            best_S = S_fine[fine_idx]
            best_theta = theta_fine[fine_idx]
        
        # Even finer: 10000 steps in a small window around the best fine theta
        if time.time() < deadline - safety_margin:
            fine_step = (math.pi / 3) / n_fine
            small_window = 2 * fine_step
            theta_very_fine = np.linspace(max(0, best_theta - small_window), min(math.pi/3, best_theta + small_window), 10000, endpoint=False)
            
            if len(theta_very_fine) > 10:
                angles_vf = theta_very_fine[:, np.newaxis] + k_angles[np.newaxis, :]
                normals_vf = np.stack([np.cos(angles_vf), np.sin(angles_vf)], axis=-1)
                
                dots_vf = np.sum(normals_vf[:, :, np.newaxis, :] * best_points[np.newaxis, np.newaxis, :, :], axis=-1)
                abs_dots_vf = np.abs(dots_vf)
                max_per_normal_vf = np.max(abs_dots_vf, axis=-1)
                max_proj_vf = np.max(max_per_normal_vf, axis=-1)
                S_vf = (2.0 / math.sqrt(3)) * max_proj_vf
                
                vf_idx = np.argmin(S_vf)
                if S_vf[vf_idx] < best_S:
                    best_S = S_vf[vf_idx]
                    best_theta = theta_very_fine[vf_idx]
    
    # Fallback
    if best_combo is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Build final center list
    final_centers = list(base_centers)
    for idx in best_combo:
        final_centers.append(ring2_centers[idx])
    
    # Format output
    inner_rows = []
    for cx, cy in final_centers:
        inner_rows.append([cx, cy, 0.0])
    
    center = (0.0, 0.0)
    side = best_S
    angle_deg = math.degrees(best_theta)
    
    return inner_rows, center, side, angle_deg
