"""Optimized packing of 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Approach: Hexagonal Lattice Initialization + Local Refinement
    # Hexagonal lattices have a higher packing density and larger minimum
    # neighbor distance than square grids for the same number of points,
    # providing a superior starting point for local optimization.
    
    def generate_hex_lattice(s):
        pts = []
        j_max = int(0.5 / (s * np.sqrt(3) / 2)) + 2
        for j in range(-j_max, j_max + 1):
            y = j * s * np.sqrt(3) / 2
            if abs(y) > 0.5:
                continue
            i_min = int(np.ceil((-0.5 - j * s / 2) / s))
            i_max = int(np.floor((0.5 - j * s / 2) / s))
            for i in range(i_min, i_max + 1):
                x = i * s + j * s / 2
                if -0.5 <= x <= 0.5:
                    pts.append((x, y))
        return np.array(pts)
    
    # Estimate optimal spacing for hex lattice
    # s ~ sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate lattice and select points
    pts = generate_hex_lattice(s)
    
    # Adjust s if necessary to get close to N points
    # If too many, we pick the closest N to center.
    # If too few, we decrease s (increase density) and retry.
    while len(pts) < N:
        s *= 0.99
        pts = generate_hex_lattice(s)
    
    if len(pts) > N:
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)
        pts = pts[indices[:N]]
        
    x0 = pts.copy()
    
    # Objective function: Soft-min of distances
    def objective(x):
        x = x.reshape(N, 2)
        x = np.clip(x, -0.5, 0.5)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        boundary = np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        )
        
        sigma = 0.001
        flat_dist = dist.flatten()
        flat_dist = flat_dist[~np.isinf(flat_dist)]
        
        all_dists = np.concatenate([flat_dist, boundary])
        
        max_d = np.max(all_dists)
        exp_terms = np.exp(-(all_dists - max_d) / sigma)
        soft_min = max_d - sigma * np.log(np.sum(exp_terms))
        
        return -soft_min
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(objective, x0.flatten(), method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    x_opt = result.x.reshape(N, 2)
    x_opt = np.clip(x_opt, -0.4999, 0.4999)
    
    return x_opt
# EVOLVE-BLOCK-END
