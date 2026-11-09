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
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])
    
    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        # Vertex angles relative to hexagon center
        vertex_angles = [math.radians(60 * k) + theta for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])
        
        # Projection of each vertex onto each normal direction
        # proj[i,j] = vx[i]*cos(normal_j) + vy[i]*sin(normal_j)
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Sample theta from 0 to 60 degrees
        for t_deg in range(0, 61, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine
        for _ in range(2):
            step = 0.1 if _ == 0 else 0.01
            for t_deg in np.arange(best_theta - step*10, best_theta + step*10 + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta
    
    # Try all subsets of size 12 from 19 centers to find best initial configuration
    from itertools import combinations
    
    best_L_init = float('inf')
    best_subset = None
    best_thetas = None
    
    # Precompute optimal theta for each center
    center_optimal = []
    for i in range(19):
        x, y, _, _ = all_centers[i]
        L_h, theta_h = optimal_theta_for_hexagon(x, y)
        center_optimal.append((L_h, theta_h))
    
    for subset in combinations(range(19), 12):
        if time.time() > deadline:
            break
        L = max(center_optimal[i][0] for i in subset)
        if L < best_L_init:
            best_L_init = L
            best_subset = subset
            best_thetas = [center_optimal[i][1] for i in subset]
    
    if best_subset is None:
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L_init = max(center_optimal[i][0] for i in best_subset)
    
    # Initial positions and rotations
    init_positions = np.array([[all_centers[i][0], all_centers[i][1]] for i in best_subset])
    init_rotations = np.array(best_thetas)
    
    # Optimization
    def objective_and_penalty(params):
        # params: [x1, y1, ..., x12, y12]
        # Rotations are fixed at init_rotations
        xs = params[0::2]
        ys = params[1::2]
        
        # Calculate L
        max_proj = -float('inf')
        for i in range(N):
            cx, cy = xs[i], ys[i]
            theta = init_rotations[i]
            L_h = min_L_for_hexagon(cx, cy, theta)
            # L_h is already scaled to side length L
            # We need to compare projections directly to avoid scaling errors in max
            # Actually min_L_for_hexagon returns L.
            # L = max_proj / (sqrt3/2). So max_proj = L * sqrt3/2.
            # We want to minimize max L.
            if L_h > max_proj:
                max_proj = L_h
        
        # Calculate Penalty for overlaps
        penalty = 0.0
        min_dist = sqrt3 # Minimum distance between centers
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        return max_proj + 10.0 * penalty
    
    # Run optimization
    if time.time() < deadline - 10.0:
        init_params = np.zeros(2*N)
        for i in range(N):
            init_params[2*i] = init_positions[i][0]
            init_params[2*i+1] = init_positions[i][1]
        
        # Use Nelder-Mead
        try:
            res = minimize(objective_and_penalty, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.success or res.fun < best_L_init:
                best_L_opt = res.fun
                # Extract positions
                opt_xs = res.x[0::2]
                opt_ys = res.x[1::2]
                
                # Verify validity
                valid = True
                # Check overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dx = opt_xs[i] - opt_xs[j]
                        dy = opt_ys[i] - opt_ys[j]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < sqrt3 - 1e-6:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    # Recalculate true L without penalty
                    true_L = 0.0
                    for i in range(N):
                        L_h = min_L_for_hexagon(opt_xs[i], opt_ys[i], init_rotations[i])
                        if L_h > true_L:
                            true_L = L_h
                    
                    if true_L < best_L_init:
                        best_L_init = true_L
                        init_positions = np.column_stack((opt_xs, opt_ys))
                        
        except Exception:
            pass
            
    # Construct inner hexagons
    inner = []
    for i in range(N):
        x, y = init_positions[i]
        theta_deg = math.degrees(init_rotations[i])
        inner.append([x, y, theta_deg])
    
    return inner, (0.0, 0.0), best_L_init, 0.0
# EVOLVE-BLOCK-END
