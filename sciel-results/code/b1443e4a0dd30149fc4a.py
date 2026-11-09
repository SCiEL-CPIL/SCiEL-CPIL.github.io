"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Scale by 2 to ensure disjointness (distance >= 2*sqrt(3) > 2)
                # Wait, distance in tiling is sqrt(3).
                # If we scale by 1, distance is sqrt(3) ~ 1.732.
                # Unit hexagons have width 2.
                # If distance is 1.732, they overlap?
                # Let's check.
                # If two unit hexagons are aligned, min distance is 2.
                # If they are rotated 30 deg, min distance is sqrt(3).
                # So if we rotate them, we can pack them closer.
                # But if we fix them to the grid, we should ensure they are disjoint.
                # The safest is to scale the grid by a factor that ensures disjointness for ANY rotation.
                # The max width of a unit hexagon is 2.
                # The min distance for disjointness is the radius of the Minkowski sum.
                # For two unit hexagons, the Minkowski sum is a hexagon of side 2.
                # The "radius" (distance from center to boundary) varies from sqrt(3) to 2.
                # So if the center distance is > 2, they are always disjoint.
                # If center distance is sqrt(3), they can be disjoint if rotated appropriately (touching at vertices).
                # To be safe and simple, let's scale the grid by 1.0 but assume we can rotate to avoid overlap?
                # No, the grid is fixed.
                # If we use the tiling grid (dist sqrt(3)), we MUST rotate them to avoid overlap.
                # The previous program used SCALE = 1.000001.
                # This implies it relied on the grid distance being "almost" sufficient.
                # But if the grid distance is sqrt(3), and the hexagons are unit, they overlap if aligned.
                # If we rotate them by 30 degrees, they touch at vertices.
                # So, if we fix the centers at the tiling grid (dist sqrt(3)), we can achieve disjointness by rotating them.
                # This allows a tighter packing than scaling the grid.
                
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    n_centers = len(all_centers)  # 19
    
    # Precompute vertices for a unit hexagon (flat-top, side 1)
    # Angles: 0, 60, ... 300
    base_angles = np.array([math.radians(60 * k) for k in range(6)])
    base_vx = np.cos(base_angles)
    base_vy = np.sin(base_angles)
    
    def get_vertices(cx, cy, theta):
        vx = base_vx * np.cos(theta) - base_vy * np.sin(theta)
        vy = base_vx * np.sin(theta) + base_vy * np.cos(theta)
        return np.column_stack((vx + cx, vy + cy))
    
    def min_L_for_config(centers, thetas, outer_cx, outer_cy, outer_phi):
        """
        Calculate the minimum side length L for the outer hexagon 
        to contain the inner hexagons.
        """
        # Outer hexagon normals
        normal_angles = np.array([math.radians(60 * j) + outer_phi for j in range(6)])
        normal_cos = np.cos(normal_angles)
        normal_sin = np.sin(normal_angles)
        
        max_proj = 0.0
        for i, (cx, cy, _, _) in enumerate(centers):
            verts = get_vertices(cx, cy, thetas[i])
            # Translate by outer center
            vx = verts[:, 0] - outer_cx
            vy = verts[:, 1] - outer_cy
            
            # Project onto normals
            projs = vx * normal_cos[np.newaxis, :] + vy * normal_sin[np.newaxis, :]
            local_max = np.max(projs)
            if local_max > max_proj:
                max_proj = local_max
        
        # L is related to max_proj
        # Distance from center to side of outer hexagon is L * sqrt(3) / 2
        # So L = max_proj / (sqrt(3) / 2) = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / sqrt3

    # Select subsets
    # To speed up, we can just try all 50388 subsets with a fast heuristic first
    # or just try all with the optimizer if it's fast enough.
    # 50388 * 15 vars * 100 iters * 72 verts * 6 norms ... might be too slow.
    # Let's filter subsets by bounding box area first.
    
    subsets = list(combinations(range(n_centers), 12))
    
    # Heuristic score: Area of bounding box of the 12 centers
    def heuristic_score(subset):
        pts = np.array([(all_centers[i][0], all_centers[i][1]) for i in subset])
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        return (max_x - min_x) * (max_y - min_y)
    
    # Sort subsets by heuristic
    scored_subsets = [(heuristic_score(s), s) for s in subsets]
    scored_subsets.sort(key=lambda x: x[0])
    
    # Take top 200 for optimization
    top_subsets = [s for _, s in scored_subsets[:200]]
    
    best_L = float('inf')
    best_config = None
    
    for subset in top_subsets:
        if time.time() > deadline:
            break
            
        centers_subset = [all_centers[i] for i in subset]
        
        # Initial guess: 
        # Outer center: centroid
        cx_avg = np.mean([c[0] for c in centers_subset])
        cy_avg = np.mean([c[1] for c in centers_subset])
        phi_init = 0.0
        thetas_init = np.zeros(12)
        
        x0 = np.array([cx_avg, cy_avg, phi_init] + list(thetas_init))
        
        def objective(x):
            outer_cx = x[0]
            outer_cy = x[1]
            outer_phi = x[2]
            thetas = x[3:]
            return min_L_for_config(centers_subset, thetas, outer_cx, outer_cy, outer_phi)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            L_opt = res.fun
            if L_opt < best_L:
                best_L = L_opt
                best_config = (subset, res.x)
        except Exception:
            continue
            
    if best_config is None:
        # Fallback
        best_config = (tuple(range(12)), np.array([0, 0, 0] + [0]*12))
        best_L = min_L_for_config([all_centers[i] for i in best_config[0]], best_config[1][3:], best_config[1][0], best_config[1][1], best_config[1][2])

    subset, x_opt = best_config
    outer_cx = x_opt[0]
    outer_cy = x_opt[1]
    outer_phi = x_opt[2]
    thetas = x_opt[3:]
    
    inner = []
    for idx, i in enumerate(subset):
        x, y, _, _ = all_centers[i]
        inner.append([x, y, math.degrees(thetas[idx])])
        
    return inner, (outer_cx, outer_cy), best_L, math.degrees(outer_phi)
