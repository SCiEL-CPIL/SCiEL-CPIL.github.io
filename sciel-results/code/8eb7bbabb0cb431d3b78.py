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
    safety_margin = 10.0

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
        
        # Max projection for each normal k
        # For a fixed k, we want max_i max_j | center_dots[i, k] + dot_verts_normals[j, k] |
        # This is equivalent to max( max_i (center_dots[i, k] + max_j dot_verts_normals[j, k]),
        #                          -min_i (center_dots[i, k] + min_j dot_verts_normals[j, k]) )
        
        # Precompute max and min vertex projections for each normal
        max_vert_proj = np.max(dot_verts_normals, axis=0)  # (6,)
        min_vert_proj = np.min(dot_verts_normals, axis=0)  # (6,)
        
        # For each normal k:
        # Max positive projection: max_i center_dots[i, k] + max_vert_proj[k]
        # Max negative projection: min_i center_dots[i, k] + min_vert_proj[k]
        
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
        
        # Calculate Penalty
        # Penalty = sum_{i < j} max(0, min_dist - dist(i, j))
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    penalty += (min_dist - dist)
                    
        return S, penalty

    def objective(params):
        centers = params.reshape(N, 2)
        S, penalty = calculate_S_and_penalty(centers)
        # We want to minimize S, but ensure penalty is 0 (or very small)
        # Use a large weight for penalty to enforce non-overlap
        return S + 1000.0 * penalty

    # Generate initial configurations
    # 1. Honeycomb cluster (Rigid body)
    # 2. Random subsets of larger lattice
    # 3. Random jittered lattice
    
    best_S = float('inf')
    best_centers = None
    best_theta = 0.0 # We will optimize positions, so theta is effectively 0 in this frame
    
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

    # Strategy:
    # 1. Take a lattice of size 2 (19 hexagons)
    # 2. Randomly select 11 of them
    # 3. Optimize the positions of these 11 using Nelder-Mead
    # 4. Repeat with different selections and initial perturbations
    
    lattice_centers = get_lattice_config(radius=2)
    
    # Pre-compute indices of all 11-subsets? Too many.
    # Instead, we can start with the "standard" 11-hexagon cluster (center + ring 1 + 4 from ring 2)
    # And then try random perturbations.
    
    # Standard 11: Center (0,0), 6 neighbors, 4 from ring 2.
    # Let's just use the first 11 from the lattice generation order (which is roughly radial)
    # Or better, construct a specific good starting point.
    
    # Let's use a few different starting points:
    starting_points = []
    
    # SP 1: Compact cluster (Center + 6 + 4)
    # The 4 from ring 2 can be chosen to be symmetric or not.
    # Let's pick 4 consecutive ones from ring 2 to start.
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = math.sqrt(3) * (q + r / 2.0)
                y = math.sqrt(3) * (math.sqrt(3) / 2.0) * r
                ring2.append([x, y])
    
    base_7 = [[0,0]] + [[math.sqrt(3)*math.cos(i*math.pi/3), math.sqrt(3)*math.sin(i*math.pi/3)] for i in range(6)]
    
    # Pick 4 from ring2. Let's pick indices 0,1,2,3 (consecutive)
    sp1 = base_7 + ring2[0:4]
    starting_points.append(np.array(sp1))
    
    # Pick 4 from ring2. Indices 0, 1, 4, 5 (symmetric)
    sp2 = base_7 + [ring2[0], ring2[1], ring2[4], ring2[5]]
    starting_points.append(np.array(sp2))
    
    # Random subsets of the 19-point lattice
    for _ in range(20):
        idx = np.random.choice(len(lattice_centers), N, replace=False)
        sp = lattice_centers[idx]
        # Add small noise
        sp = sp + np.random.uniform(-0.1, 0.1, size=(N, 2))
        starting_points.append(sp)

    # Optimization loop
    for sp in starting_points:
        if time.time() > deadline - safety_margin:
            break
            
        # Flatten for optimizer
        x0 = sp.flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000, 'maxfev': 5000})
            
            if res.fun < best_S:
                # Check if penalty is actually small
                final_centers = res.x.reshape(N, 2)
                S_val, pen_val = calculate_S_and_penalty(final_centers)
                if pen_val < 1e-3: # Valid non-overlapping
                    best_S = S_val
                    best_centers = final_centers
        except Exception:
            pass

    # If no valid result, fallback to a large box
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Refine the best solution with a tighter tolerance
    if time.time() < deadline - safety_margin:
        x0 = best_centers.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000, 'maxfev': 10000})
            if res.fun < best_S:
                final_centers = res.x.reshape(N, 2)
                S_val, pen_val = calculate_S_and_penalty(final_centers)
                if pen_val < 1e-4:
                    best_S = S_val
                    best_centers = final_centers
        except Exception:
            pass

    # Format output
    # Inner hexagons are at best_centers, angle 0
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])
    
    # Outer hexagon center is (0,0), side is best_S, angle 0
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
