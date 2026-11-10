"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Objective: maximize the minimum of (pairwise_dist/2, boundary_dist)
    # i.e. maximize r = min(min_pairwise/2, min_boundary)
    # We minimize -r.
    
    def objective(params):
        centers = params.reshape(N, 2)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        # Boundary distances
        x, y = centers[:, 0], centers[:, 1]
        boundary_dists = np.min(np.stack([0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y], axis=1), axis=1)
        min_boundary = np.min(boundary_dists)
        
        # Radius is limited by both
        r = min(min_pairwise / 2.0, min_boundary)
        
        # Penalty for points outside the square
        outside_penalty = 0.0
        for i in range(N):
            if centers[i, 0] < -0.5 or centers[i, 0] > 0.5 or centers[i, 1] < -0.5 or centers[i, 1] > 0.5:
                outside_penalty += 100.0
        
        return -r + outside_penalty
    
    # Initialization: Hexagonal lattice
    # For 50 circles in unit square, estimate spacing
    s = np.sqrt(2 / (N * np.sqrt(3)))
    
    # Generate hex grid
    nx = int(1 / s) + 4
    ny = int(1 / (s * np.sqrt(3) / 2)) + 4
    
    points = []
    for j in range(ny):
        y = (j + 0.5) * s * np.sqrt(3) / 2 - 0.5
        offset = (j % 2) * s / 2
        for i in range(nx):
            x = (i + 0.5) * s - 0.5 + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            
    points = np.array(points)
    
    # If we don't have enough points, add more with larger spacing
    if len(points) < N:
        s2 = s * 0.9
        nx2 = int(1 / s2) + 4
        ny2 = int(1 / (s2 * np.sqrt(3) / 2)) + 4
        for j in range(ny2):
            y = (j + 0.5) * s2 * np.sqrt(3) / 2 - 0.5
            offset = (j % 2) * s2 / 2
            for i in range(nx2):
                x = (i + 0.5) * s2 - 0.5 + offset
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        points = np.array(points)
    
    # Select N points closest to center
    dist_to_center = np.sum(points**2, axis=1)
    closest_indices = np.argsort(dist_to_center)[:N]
    initial_centers = points[closest_indices].copy()
    
    # Jitter to break symmetry
    np.random.seed(42)
    jitter_scale = s * 0.03
    initial_centers += np.random.normal(0, jitter_scale, initial_centers.shape)
    
    # Clip to be safely inside
    initial_centers = np.clip(initial_centers, -0.495, 0.495)
    
    x0 = initial_centers.flatten()
    
    # Multi-start Nelder-Mead with different initializations
    best_result = None
    best_score = -np.inf
    
    # Run several optimizations with different jitter seeds
    for seed in [42, 123, 456, 789, 1011]:
        np.random.seed(seed)
        centers_try = initial_centers + np.random.normal(0, jitter_scale * 2, initial_centers.shape)
        centers_try = np.clip(centers_try, -0.495, 0.495)
        
        try:
            res = minimize(objective, centers_try.flatten(), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_result = res.x.copy()
        except Exception:
            continue
    
    # Also try the original initialization
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        score = -res.fun
        if score > best_score:
            best_score = score
            best_result = res.x.copy()
    except Exception:
        pass
    
    if best_result is None:
        final_centers = initial_centers
    else:
        final_centers = best_result.reshape(N, 2)
    
    # Final clipping
    final_centers = np.clip(final_centers, -0.4999, 0.4999)
    
    return final_centers

# EVOLVE-BLOCK-END
