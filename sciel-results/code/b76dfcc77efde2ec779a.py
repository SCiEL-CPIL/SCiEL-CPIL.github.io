"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    n_centers = len(all_centers)  # 19
    
    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def min_L_for_hexagon_fast(cx, cy, theta):
        """Faster version using analytic vertex positions."""
        # Unit hexagon vertices at angles theta, theta+60, ..., theta+300
        # For a regular hexagon of side 1, vertices are at distance 1 from center
        ca = math.cos(theta)
        sa = math.sin(theta)
        
        # Precomputed cos/sin for 60-degree increments
        c60 = 0.5
        s60 = sqrt3 / 2.0
        
        # Vertex offsets relative to center
        # k=0: (cos(theta), sin(theta))
        # k=1: (cos(theta+60), sin(theta+60)) = (ca*c60 - sa*s60, ca*s60 + sa*c60)
        # k=2: (cos(theta+120), sin(theta+120)) = (-ca/2 - sa*sqrt(3)/2, ca*sqrt(3)/2 - sa/2)
        # etc.
        
        vx = np.empty(6)
        vy = np.empty(6)
        for k in range(6):
            ang = theta + k * math.pi / 3
            vx[k] = cx + math.cos(ang)
            vy[k] = cy + math.sin(ang)
        
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search: 0 to 60 degrees in 0.5 degree steps
        for t_deg in np.arange(0, 60.5, 0.5):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine with finer steps
        for step_deg in [0.1, 0.01, 0.001, 0.0001]:
            step = math.radians(step_deg)
            t_start = best_theta - step * 10
            t_end = best_theta + step * 10 + step
            for t_deg in np.arange(math.degrees(t_start), math.degrees(t_end), step_deg):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta
    
    # Minimal scale to ensure strict disjointness (distance > sqrt(3))
    SCALE = 1.000001
    
    # Precompute optimal theta for each center
    center_optimal = []
    for i in range(n_centers):
        x, y, _, _ = all_centers[i]
        L_h, theta_h = optimal_theta_for_hexagon(x, y)
        center_optimal.append((L_h, theta_h))
    
    # Try all subsets of size 12 from 19 centers
    # C(19,12) = C(19,7) = 50388
    best_L = float('inf')
    best_subset = None
    best_thetas = None
    
    for subset in combinations(range(n_centers), 12):
        if time.time() > deadline:
            break
        L = max(center_optimal[i][0] for i in subset)
        if L < best_L:
            best_L = L
            best_subset = subset
            best_thetas = [center_optimal[i][1] for i in subset]
    
    if best_subset is None:
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)
    
    # Final high-precision polish: refine thetas for the selected subset
    # to minimize the maximum L across all selected hexagons
    if time.time() < deadline - 3.0:
        polished_thetas = list(best_thetas)
        
        # First, try to reduce the max L by adjusting individual thetas
        # Use a more thorough local search for each hexagon
        for _round in range(5):
            improved = False
            current_max_L = max(min_L_for_hexagon(all_centers[i][0], all_centers[i][1], polished_thetas[idx]) 
                               for idx, i in enumerate(best_subset))
            
            for idx, i in enumerate(best_subset):
                x, y, _, _ = all_centers[i]
                # Fine-tune this hexagon's theta with a local search
                best_local_L = min_L_for_hexagon(x, y, polished_thetas[idx])
                best_local_theta = polished_thetas[idx]
                
                # Try small perturbations
                for delta_deg in np.arange(-0.1, 0.11, 0.01):
                    t_new = polished_thetas[idx] + math.radians(delta_deg)
                    L_new = min_L_for_hexagon(x, y, t_new)
                    if L_new < best_local_L:
                        best_local_L = L_new
                        best_local_theta = t_new
                
                # Even finer
                if best_local_theta != polished_thetas[idx]:
                    for delta_deg in np.arange(-0.01, 0.011, 0.001):
                        t_new = best_local_theta + math.radians(delta_deg)
                        L_new = min_L_for_hexagon(x, y, t_new)
                        if L_new < best_local_L:
                            best_local_L = L_new
                            best_local_theta = t_new
                
                if abs(best_local_theta - polished_thetas[idx]) > 1e-10:
                    polished_thetas[idx] = best_local_theta
                    improved = True
            
            if not improved:
                break
        
        best_thetas = polished_thetas
    
    # Construct inner hexagons
    inner = []
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        inner.append([x * SCALE, y * SCALE, math.degrees(best_thetas[idx])])
    
    # Adjust outer side length for the scaling
    # Recompute the actual required L with the polished thetas
    actual_max_L = 0.0
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        L_i = min_L_for_hexagon(x * SCALE, y * SCALE, best_thetas[idx])
        if L_i > actual_max_L:
            actual_max_L = L_i
    
    final_L = actual_max_L
    
    return inner, (0.0, 0.0), final_L, 0.0
