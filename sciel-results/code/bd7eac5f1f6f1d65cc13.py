import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # Hexagon geometry constants
    d = math.sqrt(3)  # distance between centers of adjacent unit hexagons
    
    # Precompute vertices relative to center (unit hexagon, pointy-top)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
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
    
    # Function to compute S for given points and theta
    def compute_S(points_arr, theta):
        k_angles = np.arange(6) * math.pi / 3
        angles = theta + k_angles
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (6, 2)
        dots = np.sum(normals[:, np.newaxis, :] * points_arr[np.newaxis, :, :], axis=-1)  # (6, N)
        abs_dots = np.abs(dots)
        max_proj = np.max(abs_dots)
        return (2.0 / math.sqrt(3)) * max_proj
    
    # All combinations of 4 from 12 ring 2 positions
    all_combinations = list(itertools.combinations(range(12), 4))
    
    best_S = float('inf')
    best_combo = None
    best_theta = 0.0
    best_points_arr = None
    
    # Phase 1: Coarse sweep over all combinations
    n_coarse = 200
    theta_coarse = np.linspace(0, math.pi / 3, n_coarse, endpoint=False)
    k_angles = np.arange(6) * math.pi / 3
    
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
        
        # Build points array for this combination
        combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])  # (66, 2)
        
        # Vectorized computation for all theta values
        angles = theta_coarse[:, np.newaxis] + k_angles[np.newaxis, :]
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (200, 6, 2)
        
        dots = np.sum(normals[:, :, np.newaxis, :] * combo_points[np.newaxis, np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        
        max_per_normal = np.max(abs_dots, axis=-1)  # (200, 6)
        max_proj = np.max(max_per_normal, axis=-1)  # (200,)
        S_vals = (2.0 / math.sqrt(3)) * max_proj
        
        min_idx = np.argmin(S_vals)
        min_S = S_vals[min_idx]
        min_theta = theta_coarse[min_idx]
        
        if min_S < best_S:
            best_S = min_S
            best_combo = combo
            best_theta = min_theta
            best_points_arr = combo_points
    
    # Phase 2: Refinement using scipy on top candidates
    if best_combo is not None and time.time() < deadline - safety_margin:
        # Get top 10 combinations by coarse score
        combo_scores = []
        for combo in all_combinations:
            if time.time() > deadline - safety_margin:
                break
            combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])
            # Quick evaluation at a few theta values
            theta_samples = np.linspace(0, math.pi / 3, 20, endpoint=False)
            best_combo_S = float('inf')
            for theta in theta_samples:
                S_val = compute_S(combo_points, theta)
                if S_val < best_combo_S:
                    best_combo_S = S_val
            combo_scores.append((best_combo_S, combo))
        
        combo_scores.sort(key=lambda x: x[0])
        top_combos = [c[1] for c in combo_scores[:10]]
        
        # Refine each top combo with scipy
        for combo in top_combos:
            if time.time() > deadline - safety_margin:
                break
            
            combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])
            
            def obj(theta):
                return compute_S(combo_points, theta)
            
            # Try multiple starting points
            best_local_S = float('inf')
            best_local_theta = 0.0
            
            n_starts = 20
            for i in range(n_starts):
                theta0 = (i / n_starts) * (math.pi / 3)
                result = minimize_scalar(obj, bounds=(0, math.pi / 3), method='bounded', 
                                        options={'xatol': 1e-10, 'maxiter': 100})
                if result.fun < best_local_S:
                    best_local_S = result.fun
                    best_local_theta = result.x
            
            if best_local_S < best_S:
                best_S = best_local_S
                best_combo = combo
                best_theta = best_local_theta
                best_points_arr = combo_points
    
    # Phase 3: Final ultra-fine refinement on the best
    if best_combo is not None and time.time() < deadline - safety_margin:
        if best_points_arr is None:
            best_points_arr = np.vstack([base_points_arr] + [ring2_points_list[i] for i in best_combo])
        
        # Fine grid around best_theta
        fine_step = (math.pi / 3) / 5000
        window = 5 * fine_step
        theta_fine = np.linspace(max(0, best_theta - window), min(math.pi/3, best_theta + window), 5000, endpoint=False)
        
        angles_fine = theta_fine[:, np.newaxis] + k_angles[np.newaxis, :]
        normals_fine = np.stack([np.cos(angles_fine), np.sin(angles_fine)], axis=-1)
        
        dots_fine = np.sum(normals_fine[:, :, np.newaxis, :] * best_points_arr[np.newaxis, np.newaxis, :, :], axis=-1)
        abs_dots_fine = np.abs(dots_fine)
        max_per_normal_fine = np.max(abs_dots_fine, axis=-1)
        max_proj_fine = np.max(max_per_normal_fine, axis=-1)
        S_fine = (2.0 / math.sqrt(3)) * max_proj_fine
        
        fine_idx = np.argmin(S_fine)
        if S_fine[fine_idx] < best_S:
            best_S = S_fine[fine_idx]
            best_theta = theta_fine[fine_idx]
    
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
