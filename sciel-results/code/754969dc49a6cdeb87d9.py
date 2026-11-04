import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Initial configuration: 4x3 block in hexagonal grid
    # Axial coordinates (q, r)
    # x = 1.5 * q
    # y = sqrt(3) / 2 * (q + 2r)
    
    q_vals = np.arange(4)
    r_vals = np.arange(3)
    
    x0 = []
    for r in r_vals:
        for q in q_vals:
            x = 1.5 * q
            y = np.sqrt(3) / 2 * (q + 2 * r)
            x0.append([x, y])
            
    # Center the configuration
    x0 = np.array(x0)
    x0 -= np.mean(x0, axis=0)
    
    # Flatten for optimizer
    x0_flat = x0.flatten()
    
    # Hexagon vertices for unit side length (flat top)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertices_rel = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def cost(x_flat):
        x = x_flat.reshape(N, 2)
        
        # Calculate required side length L
        # For each hexagon, get all vertices
        # vertices = x[:, np.newaxis, :] + vertices_rel[np.newaxis, :, :]
        
        # Unroll for speed?
        # L_req for a point (vx, vy) is max(|vx|, |vx + vy/sqrt(3)|, |-vx + vy/sqrt(3)|)
        
        # Vectorized calculation
        vx = x[:, 0]
        vy = x[:, 1]
        
        # We need to check all 6 vertices for each hexagon
        # Let's compute for all 72 points at once
        all_vx = np.repeat(vx, 6) + np.tile(vertices_rel[:, 0], N)
        all_vy = np.repeat(vy, 6) + np.tile(vertices_rel[:, 1], N)
        
        term1 = np.abs(all_vx)
        term2 = np.abs(all_vx + all_vy / np.sqrt(3))
        term3 = np.abs(-all_vx + all_vy / np.sqrt(3))
        
        L_req = np.max(np.maximum(term1, np.maximum(term2, term3)))
        
        # Calculate overlap penalty
        # Min distance between centers for non-overlap is sqrt(3)
        penalty = 0.0
        min_dist = np.sqrt(3)
        
        # Calculate pairwise distances
        # x is (12, 2)
        # dist[i, j] = ||x[i] - x[j]||
        dists = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
        
        # Only consider i < j
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        # Penalty for pairs closer than min_dist
        violations = np.maximum(0, min_dist - pair_dists)
        penalty = np.sum(violations ** 2)
        
        return L_req + 100.0 * penalty

    best_result = None
    best_cost = float('inf')
    
    # Run optimization
    # Nelder-Mead is robust for non-smooth
    result = minimize(cost, x0_flat, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if result.success or result.fun < best_cost:
        best_cost = result.fun
        best_result = result.x
        
    # If time allows, try a second run with different initial conditions or tighter tolerance
    if time.time() < deadline:
        # Try a random perturbation
        np.random.seed(42)
        x0_pert = x0_flat + np.random.normal(0, 0.5, x0_flat.size)
        result2 = minimize(cost, x0_pert, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result2.fun < best_cost:
            best_cost = result2.fun
            best_result = result2.x

    if best_result is None:
        # Fallback to initial
        best_result = x0_flat
        
    # Extract solution
    x_sol = best_result.reshape(N, 2)
    
    # Calculate final L
    vx = x_sol[:, 0]
    vy = x_sol[:, 1]
    all_vx = np.repeat(vx, 6) + np.tile(vertices_rel[:, 0], N)
    all_vy = np.repeat(vy, 6) + np.tile(vertices_rel[:, 1], N)
    term1 = np.abs(all_vx)
    term2 = np.abs(all_vx + all_vy / np.sqrt(3))
    term3 = np.abs(-all_vx + all_vy / np.sqrt(3))
    L_final = np.max(np.maximum(term1, np.maximum(term2, term3)))
    
    # Construct output
    # inner: list of [x, y, angle_degrees]
    inner = []
    for i in range(N):
        inner.append([x_sol[i, 0], x_sol[i, 1], 0.0])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = float(L_final)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
