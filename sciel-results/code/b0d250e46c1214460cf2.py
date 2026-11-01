import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal Lattice
    def hex_lattice_init(n_points):
        s = np.sqrt(np.pi / (n_points * np.sqrt(3)/2))
        k = int(1.0 / s) + 3
        
        points = []
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points.append([x, y])
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        points = points[dists < 1.0]
        
        if len(points) < n_points:
             # Fallback: scale down
             # This is unlikely for N=300
             pass
             
        indices = np.argsort(dists)
        return points[indices[:n_points]]

    centers = hex_lattice_init(N)
    
    # 2. Optimization
    n_iter = 20000
    beta_start = 10.0
    beta_end = 2000.0
    lr_start = 0.05
    lr_end = 0.0005
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
            
        progress = it / n_iter
        # Use a smooth annealing schedule
        beta = beta_start * (beta_end / beta_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(centers, axis=1)
        bound_dists = 1.0 - norms
        
        V_internal = dists / 2.0
        V_boundary = bound_dists
        
        # Global min for shifting
        # We need min of all finite values
        # V_internal has inf on diagonal
        min_int = np.min(V_internal[np.isfinite(V_internal)])
        min_bound = np.min(V_boundary)
        min_V = min(min_int, min_bound)
        
        # Compute exponentials
        exp_int = np.exp(-beta * (V_internal - min_V))
        exp_int[~np.isfinite(V_internal)] = 0.0
        S_int = np.sum(exp_int)
        
        exp_bound = np.exp(-beta * (V_boundary - min_V))
        S_bound = np.sum(exp_bound)
        
        S_total = S_int + S_bound
        
        if S_total == 0:
            break
            
        p_int = exp_int / S_total
        p_bound = exp_bound / S_total
        
        # Gradients
        dists_safe = np.where(dists > 1e-8, dists, 1.0)
        u = diff / dists_safe[:, :, np.newaxis]
        
        grad_int = np.sum(p_int[:, :, np.newaxis] * 0.5 * u, axis=1)
        
        norms_safe = np.where(norms > 1e-8, norms, 1.0)
        radial_vec = centers / norms_safe[:, np.newaxis]
        grad_bound = p_bound[:, np.newaxis] * (-radial_vec)
        
        grad = grad_int + grad_bound
        
        centers += lr * grad
        
        # Project to unit disk
        new_norms = np.linalg.norm(centers, axis=1)
        mask_out = new_norms > 1.0
        if np.any(mask_out):
            centers[mask_out] *= (1.0 / new_norms[mask_out, np.newaxis])
            
    return centers
