import numpy as np
import time
import math
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute unit hexagon vertices and edge normals
    angles = np.arange(6) * np.pi / 3
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Normals for each edge (perpendicular to edge, pointing outward)
    edge_midpoints = (unit_verts + np.roll(unit_verts, -1, axis=0)) / 2.0
    unit_normals = edge_midpoints / np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    
    def get_hex_verts(x, y, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return unit_verts @ R.T + np.array([x, y])
    
    def get_hex_normals(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return unit_normals @ R.T
    
    def sat_max_separation(v1, n1, v2, n2):
        """
        Returns the maximum separation distance between two hexagons.
        Positive means disjoint (separation > 0).
        Negative means overlapping (deepest overlap).
        """
        all_normals = np.vstack((n1, n2))
        
        proj1 = v1 @ all_normals.T
        proj2 = v2 @ all_normals.T
        
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # Separation for each axis
        sep1 = min2 - max1  # Case: hex1 is left of hex2
        sep2 = min1 - max2  # Case: hex2 is left of hex1
        overlap_depth = np.minimum(max1, max2) - np.maximum(min1, min2)
        sep3 = -overlap_depth  # Case: overlapping
        
        separations = np.maximum(sep1, np.maximum(sep2, sep3))
        
        return np.max(separations)
    
    def compute_objective(params):
        """
        Compute outer side length + penalty for overlaps.
        """
        # Compute outer side length
        max_extent = 0.0
        verts_list = []
        normals_list = []
        
        for i in range(N):
            x, y, theta = params[3*i], params[3*i+1], params[3*i+2]
            verts = get_hex_verts(x, y, theta)
            normals = get_hex_normals(theta)
            verts_list.append(verts)
            normals_list.append(normals)
            
            proj = verts @ unit_normals.T
            extents = np.max(np.abs(proj), axis=0)
            local_max = np.max(extents)
            if local_max > max_extent:
                max_extent = local_max
        
        outer_side = 2.0 * max_extent / np.sqrt(3)
        
        # Compute overlap penalties
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_max_separation(verts_list[i], normals_list[i], verts_list[j], normals_list[j])
                if sep < 0:
                    penalty += (-sep) ** 2
        
        weight = 100.0
        return outer_side + weight * penalty
    
    def initial_flower():
        params = np.zeros(3*N)
        params[0:3] = [0, 0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            x = 2.1 * np.cos(theta)
            y = 2.1 * np.sin(theta)
            params[3 + 3*i : 6 + 3*i] = [x, y, 0]
        for i in range(5):
            theta = i * 2 * np.pi / 5 + np.pi / 5
            x = 4.1 * np.cos(theta)
            y = 4.1 * np.sin(theta)
            idx = 7 + i
            params[3*idx : 3*idx+3] = [x, y, 0]
        return params
    
    def simulate_annealing(initial_params, start_temp, end_temp, steps):
        current_params = initial_params.copy()
        current_cost = compute_objective(current_params)
        best_params = current_params.copy()
        best_cost = current_cost
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Cooling
            alpha = start_temp / end_temp
            temp = end_temp * (alpha ** (step / steps))
            
            # Propose new state
            idx = np.random.randint(N)
            new_params = current_params.copy()
            
            # Move position
            delta_x = np.random.normal(0, 0.5)
            delta_y = np.random.normal(0, 0.5)
            new_params[3*idx] += delta_x
            new_params[3*idx+1] += delta_y
            
            # Rotate
            if np.random.rand() < 0.3:
                delta_theta = np.random.normal(0, 0.2)
                new_params[3*idx+2] += delta_theta
            
            new_cost = compute_objective(new_params)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or np.random.rand() < math.exp(-delta_cost / temp):
                current_params = new_params
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = current_params.copy()
                    
        return best_params, best_cost
    
    # Initial guesses
    initial_guesses = [initial_flower()]
    
    # Add some random perturbations to initial guesses for diversity
    for _ in range(3):
        p = initial_flower() + np.random.normal(0, 0.5, 3*N)
        initial_guesses.append(p)
    
    best_params_overall = None
    best_cost_overall = np.inf
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        start_sa = time.time()
        # Run SA for a while
        best_params, best_cost = simulate_annealing(guess, 10.0, 0.01, 15000)
        
        if best_cost < best_cost_overall:
            best_cost_overall = best_cost
            best_params_overall = best_params
    
    if best_params_overall is None:
        best_params_overall = initial_flower()
        best_cost_overall = compute_objective(best_params_overall)
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x, y, theta = best_params_overall[3*i], best_params_overall[3*i+1], best_params_overall[3*i+2]
        inner.append([float(x), float(y), float(np.degrees(theta))])
    
    center = (0.0, 0.0)
    side = float(best_cost_overall)
    angle = 0.0
    
    return inner, center, side, angle
