"""Rewritten program for hexagon packing."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 15.0

    # Hexagon geometry
    # Unit hexagon vertices (angle 0, pointy right)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Normals of the outer hexagon (angle 0)
    # The outer hexagon is defined by |x cos(phi) + y sin(phi)| <= S * sqrt(3)/2
    # For angle 0, normals are at 0, 60, 120, 180, 240, 300
    outer_normals = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    # Precompute dot products of hex vertices with outer normals
    # dot_verts_normals[i, k] = hex_verts[i] . outer_normals[k]
    dot_verts_normals = hex_verts @ outer_normals.T  # (6, 6)

    # Minimum distance between centers of two disjoint unit hexagons (angle 0)
    min_dist = math.sqrt(3)

    def calculate_S_and_penalty(centers):
        """
        centers: (11, 2) array of (x, y)
        Returns: (S, penalty)
        S: Side length of the bounding hexagon
        penalty: Sum of overlap penalties
        """
        # Calculate S
        # For each center C_i, and each normal N_k, and each vertex V_j:
        # Projection = (C_i + V_j) . N_k = C_i . N_k + V_j . N_k
        # We need max over i, j, k of |Projection|
        
        # C_i . N_k for all i, k -> (11, 6)
        center_dots = centers @ outer_normals.T
        
        # Precompute max and min vertex projections for each normal
        max_vert_proj = np.max(dot_verts_normals, axis=0)  # (6,)
        min_vert_proj = np.min(dot_verts_normals, axis=0)  # (6,)
        
        max_center_proj = np.max(center_dots, axis=0)  # (6,)
        min_center_proj = np.min(center_dots, axis=0)  # (6,)
        
        max_proj_vals = max_center_proj + max_vert_proj
        min_proj_vals = min_center_proj + min_vert_proj
        
        # The extent in direction k is max(max_proj_vals[k], -min_proj_vals[k])
        # Since the hexagon is symmetric, we just take the max absolute value
        extents = np.maximum(np.abs(max_proj_vals), np.abs(min_proj_vals))
        
        max_extent = np.max(extents)
        
        # S = (2 / sqrt(3)) * max_extent
        S = (2.0 / math.sqrt(3)) * max_extent
        
        # Calculate Penalty using a smoother approach
        # Penalty = sum_{i < j} max(0, min_dist - dist(i, j))^2 for smoother gradient
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
                    
        return S, penalty

    def objective(params):
        centers = params.reshape(N, 2)
        S, penalty = calculate_S_and_penalty(centers)
        # We want to minimize S, but ensure penalty is 0 (or very small)
        # Use a large weight for penalty to enforce non-overlap
        return S + 500.0 * penalty

    # Generate initial configurations
    best_S = float('inf')
    best_centers = None
    
    # Helper to generate a lattice configuration
    def get_lattice_config(radius=2):
        centers = []
        d = math.sqrt(3)
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= radius:
                    x = d * (q + r / 2.0)
                    y = d * (math.sqrt(3) / 2.0) * r
                    centers.append([x, y])
        return np.array(centers)

    lattice_centers = get_lattice_config(radius=2)
    
    # Generate diverse starting points
    starting_points = []
    
    # Ring 2 positions
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = math.sqrt(3) * (q + r / 2.0)
                y = math.sqrt(3) * (math.sqrt(3) / 2.0) * r
                ring2.append([x, y])
    
    base_7 = [[0,0]] + [[math.sqrt(3)*math.cos(i*math.pi/3), math.sqrt(3)*math.sin(i*math.pi/3)] for i in range(6)]
    
    # Various selections from ring 2
    ring2_indices = list(range(12))
    
    # Symmetric selections
    starting_points.append(np.array(base_7 + [ring2[i] for i in [0, 1, 4, 5]]))
    starting_points.append(np.array(base_7 + [ring2[i] for i in [0, 3, 6, 9]]))
    starting_points.append(np.array(base_7 + [ring2[i] for i in [1, 2, 5, 6]]))
    starting_points.append(np.array(base_7 + [ring2[i] for i in [0, 2, 4, 6]]))
    
    # Random subsets of the 19-point lattice with more diversity
    np.random.seed(42)
    for _ in range(30):
        idx = np.random.choice(len(lattice_centers), N, replace=False)
        sp = lattice_centers[idx].copy()
        # Add varied noise
        noise_scale = np.random.uniform(0.05, 0.3)
        sp = sp + np.random.uniform(-noise_scale, noise_scale, size=(N, 2))
        starting_points.append(sp)

    # Optimization loop with multiple methods
    for sp in starting_points:
        if time.time() > deadline - safety_margin:
            break
            
        x0 = sp.flatten()
        
        try:
            # Try Nelder-Mead first
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 3000, 'maxfev': 8000})
            
            if res.fun < best_S:
                final_centers = res.x.reshape(N, 2)
                S_val, pen_val = calculate_S_and_penalty(final_centers)
                if pen_val < 1e-4:
                    best_S = S_val
                    best_centers = final_centers
        except Exception:
            pass

    # If no valid result, fallback to a large box
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Multiple refinement passes with tighter tolerances
    for refinement_pass in range(3):
        if time.time() > deadline - safety_margin:
            break
            
        x0 = best_centers.flatten()
        tol = [1e-5, 1e-6, 1e-7][refinement_pass]
        maxiter = [5000, 8000, 10000][refinement_pass]
        maxfev = [10000, 15000, 20000][refinement_pass]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': tol, 'fatol': tol, 'maxiter': maxiter, 'maxfev': maxfev})
            if res.fun < best_S:
                final_centers = res.x.reshape(N, 2)
                S_val, pen_val = calculate_S_and_penalty(final_centers)
                if pen_val < 1e-5:
                    best_S = S_val
                    best_centers = final_centers
        except Exception:
            pass

    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])
    
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
