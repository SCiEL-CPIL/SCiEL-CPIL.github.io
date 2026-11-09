"""
Combined approach: Combinatorial grid search for initialization + Continuous local optimization.
"""
import math
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    # Normal directions for flat-top outer hexagon
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def get_vertices(cx, cy, theta):
        """Get 6 vertices of a unit hexagon at (cx, cy) with rotation theta."""
        # Base angles for flat-top hexagon vertices
        # 0, 60, 120, 180, 240, 300
        base_angles = np.array([math.radians(60 * k) for k in range(6)])
        angles = base_angles + theta
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        return np.column_stack((vx, vy))

    def calc_L_for_config(positions, rotations, outer_rot):
        """
        Calculate the minimum side length L of the outer hexagon (rotated by outer_rot)
        that contains all inner hexagons.
        positions: (N, 2) array
        rotations: (N,) array
        outer_rot: float
        """
        # Rotate outer normals
        cos_or = math.cos(outer_rot)
        sin_or = math.sin(outer_rot)
        
        # New normal vectors
        n_cos = normal_cos * cos_or - normal_sin * sin_or
        n_sin = normal_cos * sin_or + normal_sin * cos_or
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            theta = rotations[i]
            verts = get_vertices(cx, cy, theta)
            # Project vertices onto each normal
            projs = verts[:, 0] * n_cos + verts[:, 1] * n_sin
            # Max projection for this hexagon
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        # L = max_proj / (sqrt3 / 2)
        return max_proj / (sqrt3 / 2.0)

    def calc_penalty(positions):
        """
        Calculate penalty for overlaps.
        Returns 0 if no overlaps (distance >= 2).
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    # Quadratic penalty for overlap
                    diff = 2.0 - dist
                    penalty += diff * diff
        return penalty

    def objective(vars):
        """
        Objective function for optimizer.
        vars: [x1, y1, th1, x2, y2, th2, ..., x12, y12, th12, outer_rot]
        """
        positions = np.zeros((N, 2))
        rotations = np.zeros(N)
        for i in range(N):
            positions[i, 0] = vars[i*3]
            positions[i, 1] = vars[i*3 + 1]
            rotations[i] = vars[i*3 + 2]
        outer_rot = vars[N*3]
        
        L = calc_L_for_config(positions, rotations, outer_rot)
        pen = calc_penalty(positions)
        
        # Weight for penalty: large enough to prevent overlaps
        # If L is around 3.5, penalty of 1.0 should be significant
        return L + 10.0 * pen

    # --- Phase 1: Combinatorial Search ---
    
    # Precompute optimal theta for each grid point (independent of subset, but depends on cluster rot)
    # We will search over cluster rotation angles
    cluster_angles_to_try = []
    for a_deg in range(0, 61, 5):
        cluster_angles_to_try.append(math.radians(a_deg))
    cluster_angles_to_try.append(math.radians(30))
    
    best_L = float('inf')
    best_vars = None
    
    for cluster_angle in cluster_angles_to_try:
        if time.time() > deadline - 20.0: # Leave time for optimization
            break
            
        cos_ca = math.cos(cluster_angle)
        sin_ca = math.sin(cluster_angle)
        
        # Rotate centers
        rotated_centers = []
        for x, y, q, r in all_centers:
            rx = x * cos_ca - y * sin_ca
            ry = x * sin_ca + y * cos_ca
            rotated_centers.append((rx, ry))
            
        # Find optimal theta for each center to minimize L contribution?
        # Actually, L is global max. So we just need a good initial theta.
        # 0 is a good default.
        
        # Search subsets
        # To speed up, we can pre-sort centers by distance from origin?
        # The "worst" centers are the farthest ones.
        # We want to pick 12 centers that minimize the max extent.
        
        # Simple heuristic: Try all subsets is 19C12 = 50388. Fast enough.
        for subset in combinations(range(19), 12):
            if time.time() > deadline - 20.0:
                break
                
            positions = np.array([rotated_centers[i] for i in subset])
            rotations = np.zeros(N)
            
            # Quick check: Calculate L
            L = calc_L_for_config(positions, rotations, 0.0)
            
            if L < best_L:
                best_L = L
                # Store variables
                vars_list = []
                for i in range(N):
                    vars_list.append(positions[i, 0])
                    vars_list.append(positions[i, 1])
                    vars_list.append(0.0) # theta
                vars_list.append(0.0) # outer rot
                best_vars = vars_list

    if best_vars is None:
        # Fallback
        positions = np.array([all_centers[i][:2] for i in range(12)])
        best_vars = []
        for i in range(N):
            best_vars.append(positions[i, 0])
            best_vars.append(positions[i, 1])
            best_vars.append(0.0)
        best_vars.append(0.0)
        best_L = calc_L_for_config(positions, np.zeros(N), 0.0)

    # --- Phase 2: Continuous Optimization ---
    
    if time.time() < deadline - 5.0:
        x0 = np.array(best_vars)
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
            
            if res.success or res.fun < best_L:
                # Check if the result is valid (no overlaps)
                pos_opt = np.array(res.x[:N*3]).reshape(N, 3)
                positions = pos_opt[:, :2]
                rotations = pos_opt[:, 2]
                outer_rot = res.x[N*3]
                
                pen = calc_penalty(positions)
                if pen < 1e-3: # Very small overlap allowed
                    L_opt = calc_L_for_config(positions, rotations, outer_rot)
                    if L_opt < best_L:
                        best_L = L_opt
                        best_vars = list(res.x)
        except Exception as e:
            pass

    # --- Construct Output ---
    
    # Parse best_vars
    positions = np.zeros((N, 2))
    rotations = np.zeros(N)
    for i in range(N):
        positions[i, 0] = best_vars[i*3]
        positions[i, 1] = best_vars[i*3 + 1]
        rotations[i] = best_vars[i*3 + 2]
    outer_rot = best_vars[N*3]
    
    inner = []
    for i in range(N):
        x, y = positions[i]
        theta_deg = math.degrees(rotations[i])
        inner.append([x, y, theta_deg])
    
    return inner, (0.0, 0.0), best_L, math.degrees(outer_rot)

# EVOLVE-BLOCK-END
