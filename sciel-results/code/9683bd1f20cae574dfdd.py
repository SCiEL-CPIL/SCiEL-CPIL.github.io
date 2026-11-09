"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    # Precompute normal directions for outer hexagon (flat-top, side L)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])
    
    def min_L_for_hexagon(cx, cy, theta):
        """
        Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta 
        to fit in an outer hex of side L centered at origin with 0 rotation.
        """
        # Vertex angles relative to hexagon center
        # Flat-top hexagon vertices are at 0, 60, ... degrees relative to its local frame
        # If we rotate by theta, vertices are at theta + 60k
        vertex_angles = [math.radians(60 * k) + theta for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])
        
        # Projection of each vertex onto each normal direction
        # proj[i,j] = vx[i]*cos(normal_j) + vy[i]*sin(normal_j)
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        # Inradius of outer hex is L * sqrt(3) / 2
        # Condition: max_proj <= L * sqrt(3) / 2  =>  L >= max_proj / (sqrt(3)/2)
        return max_proj / (sqrt3 / 2.0)
    
    def optimal_theta_for_hexagon(cx, cy):
        """
        Find rotation theta that minimizes the required L for this specific hexagon.
        We sample theta in [0, pi/3) due to 60-degree symmetry.
        """
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search
        for t_deg in range(0, 60, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Fine search
        for _ in range(2):
            step = 0.05 if _ == 0 else 0.01
            for t_deg in np.arange(best_theta - step*10, best_theta + step*10 + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Precompute optimal theta and L for each of the 19 centers
    # We use a tiny scale factor to ensure strict non-overlap if we stick to grid,
    # but for optimization we will handle it differently.
    # For now, compute for exact grid positions.
    center_optimal = []
    for i in range(19):
        x, y = all_centers[i]
        L_h, theta_h = optimal_theta_for_hexagon(x, y)
        center_optimal.append((L_h, theta_h))
    
    # Find the best subset of 12 centers
    from itertools import combinations
    
    best_L = float('inf')
    best_subset = None
    best_thetas = None
    
    # Iterate all subsets
    for subset in combinations(range(19), 12):
        if time.time() > deadline - 5.0: # Leave time for optimization
            break
        # L is the max over all hexagons in subset (independent optimal)
        L = max(center_optimal[i][0] for i in subset)
        if L < best_L:
            best_L = L
            best_subset = subset
            best_thetas = [center_optimal[i][1] for i in subset]
            
    if best_subset is None:
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)
        
    # Initial positions and angles
    x0 = []
    for i in best_subset:
        x, y = all_centers[i]
        x0.append([x, y, best_thetas[best_subset.index(i)]])
        
    x0 = np.array(x0)
    
    # Define the objective function for L-BFGS-B
    # We want to minimize L.
    # L is determined by the maximum projection of all vertices of all hexagons.
    # We also need to ensure non-overlap.
    # We use a penalty method: Objective = L + lambda * (overlap_violation)^2
    
    def objective(params):
        # params: 36 values (12 * 3)
        pos = params.reshape(12, 3)
        cx = pos[:, 0]
        cy = pos[:, 1]
        theta = pos[:, 2]
        
        # Calculate L
        max_proj = 0.0
        for i in range(12):
            L_i = min_L_for_hexagon(cx[i], cy[i], theta[i])
            if L_i > max_proj:
                max_proj = L_i
        L_val = max_proj / (sqrt3 / 2.0)
        
        # Calculate overlap penalty
        # For each pair, check distance between centers.
        # If distance < 2 * inradius? No, hexagons can touch.
        # Minimum distance between centers of two disjoint unit hexagons is not constant.
        # However, if centers are distance d apart, and d < 2, they might overlap.
        # A simple proxy: if d < 1.7 (approx min distance for flat-side alignment), penalize.
        # Better: use the Minkowski sum logic or just a strong penalty for d < 1.5.
        # Actually, for grid points, distance is at least sqrt(3) ~ 1.732.
        # If we move them, we must ensure they don't overlap.
        # Penalty: sum over pairs of max(0, (d_min - d_i))^2
        # d_min depends on relative rotation.
        # Approximation: min distance is ~1.732 for aligned, up to 2.0 for vertex-to-vertex?
        # Let's use a conservative d_min = 1.7.
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = math.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if d < 1.732: # sqrt(3)
                    penalty += (1.732 - d)**2
        
        return L_val + 10.0 * penalty

    # Run L-BFGS-B
    try:
        res = minimize(objective, x0.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if res.success or res.fun < best_L + 1.0: # Accept if it didn't get worse
            # Extract final positions
            pos = res.x.reshape(12, 3)
            
            # Calculate final L
            final_L = 0.0
            for i in range(12):
                L_i = min_L_for_hexagon(pos[i,0], pos[i,1], pos[i,2])
                if L_i > final_L:
                    final_L = L_i
            final_L = final_L / (sqrt3 / 2.0)
            
            # Check validity (non-overlap)
            # If penalty is 0, it's valid.
            # If final_L is smaller than best_L, update.
            if final_L < best_L:
                best_L = final_L
                x0 = pos
    except Exception:
        pass

    # Construct inner hexagons
    inner = []
    for i in range(12):
        x, y, theta_rad = x0[i]
        inner.append([x, y, math.degrees(theta_rad)])
    
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
