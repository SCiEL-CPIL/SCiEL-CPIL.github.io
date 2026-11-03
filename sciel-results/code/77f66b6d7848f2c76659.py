import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 115

def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _potential_energy(points, eps=1e-6):
    """
    Compute a smooth potential energy that encourages spheres to be far apart
    and away from walls. Minimizing this energy approximates maximizing the
    minimum distance.
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, eps**2)
    
    # Sum of inverse squared distances for all pairs
    # We only sum upper triangle to avoid double counting, but for optimization
    # symmetry doesn't matter as long as it's consistent.
    iu = np.triu_indices_from(dists_sq, k=1)
    pair_energy = np.sum(1.0 / dists_sq[iu])
    
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    wall_dists = np.maximum(wall_dists, eps)
    wall_energy = np.sum(1.0 / (wall_dists ** 2))
    
    return pair_energy + wall_energy

def _potential_energy_grad(points, eps=1e-6):
    """Gradient of the potential energy."""
    grad = np.zeros_like(points)
    
    # Pairwise gradient
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists_sq = np.maximum(dists_sq, eps**2)
    
    # Gradient of sum_{i<j} 1/d^2 w.r.t x_i is -2 * (x_i - x_j) / d^4
    # Let's compute the force vector.
    # F_ij = -2 * (x_i - x_j) / ||x_i - x_j||^4
    # This is attractive/repulsive.
    
    # Efficient calculation:
    # inv_d4 = 1 / (dists_sq ** 2)
    # grad_i = -2 * sum_{j!=i} (x_i - x_j) * inv_d4[i,j]
    
    inv_d4 = 1.0 / (dists_sq ** 2)
    np.fill_diagonal(inv_d4, 0.0)
    
    # grad = -2 * (points * sum(inv_d4, axis=1)[:, None] - (diff * inv_d4[:, :, None]).sum(axis=1))
    # Wait, diff[i,j] = x_i - x_j
    # Sum over j of (x_i - x_j) / d_ij^4
    # = x_i * sum(1/d^4) - sum(x_j / d^4)
    
    sum_inv_d4 = np.sum(inv_d4, axis=1)
    sum_x_j_inv_d4 = np.sum(diff * inv_d4[:, :, None], axis=1)
    
    grad_pairs = -2.0 * (points * sum_inv_d4[:, None] - sum_x_j_inv_d4)
    grad += grad_pairs
    
    # Wall gradient
    # Energy = sum 1/w^2 where w = 0.5 - |x|
    # dE/dx = -2 * (1/w^3) * dw/dx
    # dw/dx = -sign(x)
    # dE/dx = 2 * sign(x) / w^3
    
    wall_dists = 0.5 - np.abs(points)
    wall_dists = np.maximum(wall_dists, eps)
    signs = np.sign(points)
    # If x is 0, sign is 0, but derivative is undefined. We can ignore or handle carefully.
    # For optimization, 0 is fine as it's a measure zero set.
    grad_walls = 2.0 * signs / (wall_dists ** 3)
    grad += grad_walls
    
    return grad

def _initial_grid(n):
    """Create an initial grid-like configuration."""
    m = int(np.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.permutation(len(grid))
    return grid[idx[:n]].copy()

def _optimize_one(points, max_time):
    """Optimize a single configuration using L-BFGS-B."""
    start_time = time.time()
    
    # Flatten points for scipy
    x0 = points.flatten()
    
    # Define objective and gradient for flat vector
    def obj_flat(x):
        pts = x.reshape(-1, 3)
        return _potential_energy(pts)
        
    def grad_flat(x):
        pts = x.reshape(-1, 3)
        return _potential_energy_grad(pts).flatten()

    # Bounds: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use L-BFGS-B
    # maxiter is a soft limit, we also check time
    result = minimize(
        obj_flat, 
        x0, 
        method='L-BFGS-B', 
        jac=grad_flat, 
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 10000}
    )
    
    best_points = result.x.reshape(-1, 3)
    best_score = _score(best_points)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > max_time:
        # Return best so far (which is the result of minimize)
        pass
        
    return best_points, best_score

def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Strategy: Multiple restarts with different initializations
    # 1. Grid
    # 2. Random
    # 3. Random with larger variance
    
    n_restarts = 0
    while time.time() - start_time < DEADLINE_SECONDS:
        n_restarts += 1
        remaining_time = DEADLINE_SECONDS - (time.time() - start_time)
        
        # Choose initialization strategy
        if n_restarts == 1:
            points = _initial_grid(N)
            points += np.random.randn(N, 3) * 0.05
        elif n_restarts % 2 == 0:
            points = np.random.uniform(-0.45, 0.45, (N, 3))
        else:
            # Clustered random?
            points = np.random.uniform(-0.4, 0.4, (N, 3))
            points += np.random.randn(N, 3) * 0.1
            
        points = np.clip(points, -0.5, 0.5)
        
        # Optimize
        opt_points, opt_score = _optimize_one(points, remaining_time)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
            
    if best_points is None:
        best_points = _initial_grid(N)
        
    return best_points

if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
