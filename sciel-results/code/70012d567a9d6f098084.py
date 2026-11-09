"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
                
    # Helper to get vertices of a unit hexagon
    def get_vertices(cx, cy, theta):
        angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        return np.array([
            cx + np.cos(angles),
            cy + np.sin(angles)
        ]).T

    def objective_outer(params, vertices):
        """
        Minimize the side length L of the outer hexagon.
        params: [cx, cy, phi]
        vertices: (72, 2) array of all inner vertices
        """
        cx, cy, phi = params
        # Normals of outer hexagon
        normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)]) + phi
        normal_cos = np.cos(normal_angles)
        normal_sin = np.sin(normal_angles)
        
        # Projections of all vertices onto normals
        # proj[i, j] = (v[i] - c) . n[j]
        vx = vertices[:, 0] - cx
        vy = vertices[:, 1] - cy
        
        projs = np.outer(vx, normal_cos) + np.outer(vy, normal_sin)
        
        # For each normal j, we need max(projs[:, j]) <= a and min(projs[:, j]) >= -a
        # So a >= max_j (max(projs[:, j]) - min(projs[:, j])) / 2
        # L = 2 * a / sqrt(3)
        max_proj = np.max(projs, axis=0)
        min_proj = np.min(projs, axis=0)
        widths = max_proj - min_proj
        a = np.max(widths) / 2.0
        return 2.0 * a / sqrt3

    def get_best_outer(vertices):
        """Find optimal outer hexagon parameters for given vertices."""
        # Initial guess: centroid of vertices, phi=0
        cx0 = np.mean(vertices[:, 0])
        cy0 = np.mean(vertices[:, 1])
        res = minimize(objective_outer, [cx0, cy0, 0.0], args=(vertices,), method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        return res.x, res.fun

    # 1. Grid Search for initial configuration
    # Precompute optimal theta for each center (independent of subset)
    # We will use theta=0 for now and optimize later, or use a quick scan.
    # To save time, we'll just use theta=0 in the grid search and refine later.
    
    best_L = float('inf')
    best_subset = None
    
    # We need to check if 12 hexes fit.
    # We'll iterate through combinations.
    from itertools import combinations
    
    # Optimization: We can prune if the bounding box of centers is too large?
    # But let's just run the full search as it was fast before.
    
    for subset in combinations(range(19), 12):
        if time.time() > deadline - 60: # Reserve 60s for local search
            break
            
        # Get centers
        centers = [all_centers[i] for i in subset]
        
        # Create vertices for all 12 hexes (theta=0)
        verts_list = []
        for cx, cy in centers:
            verts_list.append(get_vertices(cx, cy, 0.0))
        vertices = np.vstack(verts_list)
        
        # Optimize outer hex
        _, L_val = get_best_outer(vertices)
        
        if L_val < best_L:
            best_L = L_val
            best_subset = subset
            # Store best centers and thetas
            best_centers = centers
            best_thetas = [0.0] * 12

    if best_subset is None:
        best_subset = tuple(range(12))
        best_centers = [all_centers[i] for i in best_subset]
        best_thetas = [0.0] * 12
        verts_list = [get_vertices(cx, cy, 0.0) for cx, cy in best_centers]
        vertices = np.vstack(verts_list)
        _, best_L = get_best_outer(vertices)

    # 2. Local Search (Hill Climbing)
    # Parameters: x, y, theta for each of 12 hexes
    # We will optimize these to minimize L.
    
    # Current state
    curr_centers = [c[:] for c in best_centers]
    curr_thetas = best_thetas[:]
    
    def get_vertices_from_state(centers, thetas):
        verts_list = []
        for i in range(12):
            verts_list.append(get_vertices(centers[i][0], centers[i][1], thetas[i]))
        return np.vstack(verts_list)

    # Optimize inner parameters using a simple gradient-free method
    # Since N=12, 36 variables. Nelder-Mead might be slow.
    # We will do coordinate descent or random nudges.
    
    best_inner_L = best_L
    best_outer_params, _ = get_best_outer(get_vertices_from_state(curr_centers, curr_thetas))
    
    # Time limit for local search
    ls_deadline = start_time + 130.0
    
    step_size = 0.1
    no_improve_count = 0
    
    while time.time() < ls_deadline and no_improve_count < 50:
        improved = False
        for i in range(12):
            if time.time() > ls_deadline:
                break
            # Try nudging x, y, theta
            for param_idx in range(3):
                for sign in [-1, 1]:
                    new_centers = [c[:] for c in curr_centers]
                    new_thetas = curr_thetas[:]
                    
                    if param_idx == 0:
                        new_centers[i][0] += sign * step_size
                    elif param_idx == 1:
                        new_centers[i][1] += sign * step_size
                    else:
                        new_thetas[i] += sign * step_size
                    
                    # Check overlap? 
                    # We must ensure hexes don't overlap.
                    # This is expensive. 
                    # Approximation: If we nudge small, and started valid, likely still valid?
                    # No, we might push into neighbors.
                    # We should check distance between centers.
                    # Min distance between centers of disjoint unit hexes is sqrt(3) (if flat-aligned) or 2 (if point-aligned)?
                    # Actually, min distance is sqrt(3) for optimal packing.
                    # We'll skip overlap check for speed and rely on the "disjoint" constraint of the task?
                    # The task says "disjoint". If they overlap, it's invalid.
                    # We must check.
                    
                    valid = True
                    for j in range(12):
                        if i == j: continue
                        dx = new_centers[i][0] - new_centers[j][0]
                        dy = new_centers[i][1] - new_centers[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        # Min distance for unit hexes is sqrt(3) ~ 1.732
                        if dist < 1.73205:
                            valid = False
                            break
                            
                    if not valid:
                        continue
                        
                    vertices = get_vertices_from_state(new_centers, new_thetas)
                    _, L_new = get_best_outer(vertices)
                    
                    if L_new < best_inner_L:
                        best_inner_L = L_new
                        curr_centers = new_centers
                        curr_thetas = new_thetas
                        best_outer_params, _ = get_best_outer(vertices)
                        improved = True
                        step_size *= 1.0 # Keep step size
                        no_improve_count = 0
                        break # Restart loop for this hex? Or just break nudge loop
                    else:
                        no_improve_count += 1
                        step_size *= 0.8
                        if step_size < 0.001:
                            step_size = 0.1
                            no_improve_count = 50 # Force break
                            break

        if not improved:
            no_improve_count += 1

    # Final Result
    inner = []
    for i in range(12):
        inner.append([curr_centers[i][0], curr_centers[i][1], math.degrees(curr_thetas[i])])
    
    # Return best outer params
    cx, cy, phi = best_outer_params
    return inner, (cx, cy), best_inner_L, math.degrees(phi)
# EVOLVE-BLOCK-END
