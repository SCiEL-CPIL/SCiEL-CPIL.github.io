import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # Hexagon geometry constants
    d = math.sqrt(3)
    
    # Generate centers for 1, 6, 12 rings
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 12 neighbors
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
                
    if len(ring2_centers) != 12:
        # Fallback logic if generation fails, though it should be 12
        pass

    # Precompute vertices for a unit hexagon (pointy-top)
    # Vertices relative to center
    verts_rel = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    # Precompute the base points for the first 7 centers (center + ring 1)
    base_centers = np.array(centers[:7])
    base_points = (base_centers[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]).reshape(-1, 2)
    
    # Precompute ring 2 center points
    ring2_centers_arr = np.array(ring2_centers)
    ring2_points = (ring2_centers_arr[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]).reshape(-1, 2)
    
    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    
    # Use symmetry to reduce combinations. The 12 ring-2 centers form a hexagonal ring.
    # Under 60-degree rotation, they permute. We can fix one center's position and 
    # only consider combinations that include center 0, then account for rotations.
    # Actually, let's just enumerate all 495 but use a faster evaluation.
    
    all_combinations = list(itertools.combinations(range(12), 4))
    
    # Precompute all possible point sets for each combination
    # For each combo, points = base_points + ring2_points for those 4 centers
    
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
            
        # Construct the 11 centers
        combo_indices = np.array(combo)
        combo_points = ring2_points[combo_indices * 6 : (combo_indices + 1) * 6]
        points = np.vstack([base_points, combo_points])
        
        # Function to compute max projection for a given theta
        def calc_S(theta):
            # Normals at theta + k*pi/3
            angles = theta + np.arange(6) * (math.pi / 3)
            normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (6, 2)
            
            # Projections: (66, 6)
            projs = points @ normals.T
            max_proj = np.max(np.abs(projs))
            return (2.0 / math.sqrt(3)) * max_proj

        # Coarse sweep to find a good starting point for local optimization
        coarse_thetas = np.linspace(0, math.pi / 3, 100)
        coarse_Ss = [calc_S(t) for t in coarse_thetas]
        min_idx = np.argmin(coarse_Ss)
        best_coarse_theta = coarse_thetas[min_idx]
        best_coarse_S = coarse_Ss[min_idx]
        
        # Local refinement using minimize_scalar
        try:
            res = minimize_scalar(calc_S, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_coarse_S:
                best_local_S = res.fun
                best_local_theta = res.x
            else:
                best_local_S = best_coarse_S
                best_local_theta = best_coarse_theta
        except Exception:
            best_local_S = best_coarse_S
            best_local_theta = best_coarse_theta
            
        if best_local_S < best_S:
            best_S = best_local_S
            best_config = combo
            best_theta = best_local_theta

    if best_config is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner_rows = []
    for cx, cy in centers[:7]:
        inner_rows.append([float(cx), float(cy), 0.0])
    for idx in best_config:
        cx, cy = ring2_centers[idx]
        inner_rows.append([float(cx), float(cy), 0.0])
        
    center = (0.0, 0.0)
    side = float(best_S)
    angle_deg = float(math.degrees(best_theta))
    
    return inner_rows, center, side, angle_deg
# EVOLVE-BLOCK-END
