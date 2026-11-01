"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sys

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin before 120s limit
    
    # Initial hexagonal packing
    def hex_init(n, square_size=1.0):
        # Pack circles in a hexagonal lattice inside [-0.5, 0.5]^2
        # Start with a reasonable estimate
        # For n circles in a square, approximate spacing
        cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        rows = int(np.ceil(n / cols))
        
        # Hexagonal spacing
        dx = square_size / (cols + 0.5)
        dy = dx * np.sqrt(3) / 2
        
        centers = []
        for i in range(rows):
            for j in range(cols):
                x = -square_size/2 + dx/2 + j * dx + (i % 2) * dx/2
                y = -square_size/2 + dy/2 + i * dy
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    centers.append([x, y])
                if len(centers) >= n:
                    break
            if len(centers) >= n:
                break
        
        centers = np.array(centers[:n])
        return centers
    
    # Compute score: largest radius such that all circles are disjoint and inside square
    def compute_radius(centers):
        n = len(centers)
        # Min distance between any two centers
        if n < 2:
            min_dist = np.inf
        else:
            dists = cdist(centers, centers)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
        
        # Min distance to boundary
        # For each center, distance to each of the 4 walls
        dist_to_walls = np.min([
            centers[:, 0] + 0.5,  # left wall
            0.5 - centers[:, 0],  # right wall
            centers[:, 1] + 0.5,  # bottom wall
            0.5 - centers[:, 1]   # top wall
        ], axis=0)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_dist / 2.0, min_wall_dist)
    
    # Objective function for optimization: maximize radius = minimize negative radius
    def objective(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        r = compute_radius(centers)
        return -r
    
    # Gradient of objective (approximate)
    def objective_grad(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)
        eps = 1e-7
        grad = np.zeros_like(centers_flat)
        
        for i in range(n):
            for j in range(2):
                old_val = centers_flat[i*2+j]
                centers_flat[i*2+j] = old_val + eps
                f_plus = objective(centers_flat)
                centers_flat[i*2+j] = old_val - eps
                f_minus = objective(centers_flat)
                centers_flat[i*2+j] = old_val
                grad[i*2+j] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    # Better gradient using analytical approach
    def objective_grad_analytical(centers_flat):
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)
        grad = np.zeros_like(centers_flat)
        
        # Compute all pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum distance pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_dist = dists[min_idx]
        
        # Compute distance to walls for each center
        dist_to_walls = np.zeros((n, 4))
        dist_to_walls[:, 0] = centers[:, 0] + 0.5  # left
        dist_to_walls[:, 1] = 0.5 - centers[:, 0]  # right
        dist_to_walls[:, 2] = centers[:, 1] + 0.5  # bottom
        dist_to_walls[:, 3] = 0.5 - centers[:, 1]  # top
        
        min_wall_idx = np.argmin(dist_to_walls, axis=1)
        min_wall_dist = np.min(dist_to_walls, axis=1)
        global_min_wall = np.min(min_wall_dist)
        global_min_wall_center = np.argmin(min_wall_dist)
        
        # The radius is min(min_dist/2, min_wall_dist)
        # Gradient with respect to centers
        # For the minimum distance pair, the gradient pushes them apart
        if min_dist / 2.0 < global_min_wall:
            # Minimum is from pairwise distance
            i, j = min_idx
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                diff_norm = diff / norm
                grad[i*2:(i+1)*2] = -diff_norm  # push i away from j
                grad[j*2:(j+1)*2] = diff_norm   # push j away from i
        else:
            # Minimum is from wall distance
            ci = global_min_wall_center
            wall = min_wall_idx[ci]
            if wall == 0:  # left wall
                grad[ci*2:(ci+1)*2] = [-1, 0]
            elif wall == 1:  # right wall
                grad[ci*2:(ci+1)*2] = [1, 0]
            elif wall == 2:  # bottom wall
                grad[ci*2:(ci+1)*2] = [0, -1]
            else:  # top wall
                grad[ci*2:(ci+1)*2] = [0, 1]
        
        return grad
    
    # Simulated annealing with cluster moves
    def simulated_annealing(centers, max_iters=50000, t_start=0.01, t_end=0.0001):
        n = len(centers)
        current_radius = compute_radius(centers)
        best_centers = centers.copy()
        best_radius = current_radius
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (it / max_iters)
            
            # Choose move type: single or cluster
            if np.random.random() < 0.3:
                # Cluster move: move a small group of nearby circles
                center_idx = np.random.randint(n)
                # Find nearby circles
                dists_to_center = np.linalg.norm(centers - centers[center_idx], axis=1)
                # Pick 2-4 nearby circles
                num_in_cluster = np.random.randint(2, 5)
                nearby = np.argsort(dists_to_center)[:num_in_cluster + 1]
                nearby = np.delete(nearby, np.where(nearby == center_idx)[0])
                nearby = np.concatenate([[center_idx], nearby])
                
                # Move cluster
                move = np.random.randn(2) * 0.02 * t
                new_centers = centers.copy()
                for idx in nearby:
                    new_centers[idx] += move
            else:
                # Single circle move
                idx = np.random.randint(n)
                move = np.random.randn(2) * 0.05 * t
                new_centers = centers.copy()
                new_centers[idx] += move
            
            # Check if new centers are valid (inside square)
            if np.any(new_centers < -0.5) or np.any(new_centers > 0.5):
                continue
            
            new_radius = compute_radius(new_centers)
            
            # Accept if better or with probability exp((new-old)/t)
            delta = new_radius - current_radius
            if delta > 0 or np.random.random() < np.exp(delta / max(t, 1e-10)):
                centers = new_centers
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
        
        return best_centers, best_radius
    
    # Start with hexagonal initialization
    centers = hex_init(N)
    current_radius = compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Phase 1: Simulated annealing to find good region
    centers, radius = simulated_annealing(centers, max_iters=20000, t_start=0.02, t_end=0.001)
    if radius > best_radius:
        best_radius = radius
        best_centers = centers.copy()
    
    if time.time() > deadline - 20:
        return best_centers
    
    # Phase 2: Local optimization using scipy
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    centers_flat = best_centers.flatten()
    
    # Try multiple restarts from different initializations
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            init = best_centers.copy()
        else:
            # Add small perturbation
            init = best_centers + np.random.randn(N, 2) * 0.01
        
        # Clip to valid range
        init = np.clip(init, -0.4999, 0.4999)
        
        try:
            result = minimize(
                objective,
                init.flatten(),
                jac=objective_grad_analytical,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_centers = result.x.reshape(-1, 2)
            new_radius = compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        except Exception as e:
            pass
    
    if time.time() > deadline - 5:
        return best_centers
    
    # Phase 3: High-precision polish
    # Use very tight tolerances
    centers_flat = best_centers.flatten()
    try:
        result = minimize(
            objective,
            centers_flat,
            jac=objective_grad_analytical,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        new_centers = result.x.reshape(-1, 2)
        new_radius = compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
    except Exception as e:
        pass
    
    return best_centers
# EVOLVE-BLOCK-END
