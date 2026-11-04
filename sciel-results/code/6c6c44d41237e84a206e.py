"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from itertools import combinations
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    
    # Unit hexagon vertices (circumradius 1)
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    base_hex_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Lattice vectors for hex tiling (spacing sqrt(3))
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    def h_hex(angle):
        # Support function of unit hexagon
        n = np.array([np.cos(angle), np.sin(angle)])
        return np.max(np.sum(base_hex_verts * n, axis=1))
    
    # Generate base 7-point hexagon
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0: continue
            c = i*u + j*v
            if np.linalg.norm(c) < 1.8:
                neighbors.append(c)
    if len(neighbors) != 6:
        neighbors = [u, -u, v, -v, u-v, v-u]
    base_7 = np.vstack([np.array([[0.0, 0.0]]), np.array(neighbors)])
    
    # Generate perimeter candidates
    perim_candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            c = i*u + j*v
            d = np.linalg.norm(c)
            if 1.6 < d < 3.5:
                if not any(np.allclose(c, b) for b in base_7):
                    perim_candidates.append(c)
    perim_candidates = np.array(perim_candidates)
    # Sort by distance to origin and take 12 closest
    perim_candidates = perim_candidates[np.argsort([np.linalg.norm(c) for c in perim_candidates])]
    perim_candidates = perim_candidates[:12]
    
    best_S = float('inf')
    best_set = None
    best_t_final = np.array([0.0, 0.0])
    
    # Search combinations of 4 points from the 12 perimeter candidates
    for comb in combinations(range(len(perim_candidates)), 4):
        new_points = perim_candidates[list(comb)]
        full_set = np.vstack([base_7, new_points])
        centroid = np.mean(full_set, axis=0)
        
        best_A_for_shape = float('inf')
        
        # Scan rotation
        for phi in np.linspace(0, np.pi/3, 120):
            n_normals = np.array([
                np.array([np.cos(phi + k * np.pi/3), np.sin(phi + k * np.pi/3)])
                for k in range(6)
            ])
            
            dots = full_set @ n_normals.T
            M = np.max(dots, axis=0) + np.array([h_hex(phi + k * np.pi/3) for k in range(6)])
            
            # Optimize translation t
            # We want to minimize max_k (M_k + t @ n_k)
            # Start with t = -centroid
            # Do a small local search
            best_t = -centroid
            best_cost = np.max(M + best_t @ n_normals.T)
            
            # Small steps in t
            for step_x in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                for step_y in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                    t_trial = -centroid + np.array([step_x, step_y])
                    cost = np.max(M + t_trial @ n_normals.T)
                    if cost < best_cost:
                        best_cost = cost
                        best_t = t_trial
            
            if best_cost < best_A_for_shape:
                best_A_for_shape = best_cost
                best_phi = phi
                best_t_final = best_t
                
        S = 2 * best_A_for_shape / np.sqrt(3)
        if S < best_S:
            best_S = S
            best_set = full_set
            
    final_centers = best_set + best_t_final
    
    if time.time() > deadline - 5:
        return [[c[0], c[1], 0.0] for c in final_centers], (0.0, 0.0), best_S, 0.0
        
    return [[c[0], c[1], 0.0] for c in final_centers], (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
