"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100
BOX_MIN = -0.5
BOX_MAX = 0.5
BOUNDS = [(BOX_MIN, BOX_MAX)] * (N * 3)

def _compute_radius(centers):
    """
    Compute the maximum radius for given centers.
    r = min( min_dist_to_wall, min_pair_dist / 2 )
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    # centers shape (N, 3)
    # wall_dists[i, k] = 0.5 - |centers[i, k]|
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    # For N=100, cKDTree is fast
    try:
        tree = cKDTree(centers)
        distances, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(distances[:, 1])
    except Exception:
        return 0.0

    return min(min_wall, min_pair_dist / 2.0)

def _repulsion_energy(flat_centers):
    """
    Objective function to minimize.
    We want to maximize repulsion, so we minimize negative repulsion.
    Using power 4 to emphasize close pairs.
    """
    centers = flat_centers.reshape(N, 3)
    
    # Calculate pairwise distances
    # To avoid division by zero, add epsilon
    # We can compute this efficiently
    # diff[i, j] = centers[i] - centers[j]
    # dist_sq[i, j] = sum(diff^2)
    
    # For N=100, 100x100 matrix is fine
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dist_sq, 1.0) # Avoid div by 0
    
    # Sum of 1/dist^4
    # dist^4 = dist_sq^2
    energy = np.sum(1.0 / (dist_sq ** 2))
    
    return -energy

def _repulsion_grad(flat_centers):
    """
    Gradient of the repulsion energy.
    E = - sum_{i<j} 1/d_ij^4 = - 0.5 * sum_{i!=j} 1/d_ij^4
    dE/dx_i = - sum_{j!=i} d/dx_i (1/d_ij^4)
    d/dx_i (1/d_ij^4) = -4 * d_ij^-5 * d(d_ij)/dx_i
    d(d_ij)/dx_i = (x_i - x_j) / d_ij
    So d/dx_i (1/d_ij^4) = -4 * (x_i - x_j) / d_ij^6
    
    So dE/dx_i = - sum_{j!=i} (-4 * (x_i - x_j) / d_ij^6)
               = 4 * sum_{j!=i} (x_i - x_j) / d_ij^6
    """
    centers = flat_centers.reshape(N, 3)
    grad = np.zeros_like(centers)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # We need 1/d^6 = 1/(dist_sq^3)
    inv_d6 = 1.0 / (dist_sq ** 3)
    np.fill_diagonal(inv_d6, 0.0)
    
    # grad_i = 4 * sum_j (x_i - x_j) * inv_d6[i,j]
    # diff[i,j] is x_i - x_j
    # So grad = 4 * sum(diff * inv_d6[:, :, np.newaxis], axis=1)
    grad = 4.0 * np.sum(diff * inv_d6[:, :, np.newaxis], axis=1)
    
    return grad.flatten()

def _generate_initial_points(method='random', seed=None):
    """
    Generate initial points.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    if method == 'random':
        # Random uniform in [-0.4, 0.4] to stay safely inside
        return rng.uniform(-0.4, 0.4, (N, 3))
    elif method == 'grid_jitter':
        # Create a grid and add jitter
        # 5x4x5 = 100 points
        # Let's do a 5x5x4 grid = 100
        # Spacing: 1.0 / 5 = 0.2
        # Points at -0.4, -0.2, 0.0, 0.2, 0.4
        # Z: -0.3, -0.1, 0.1, 0.3 (spacing 0.2, 4 points)
        x = np.linspace(-0.4, 0.4, 5)
        y = np.linspace(-0.4, 0.4, 5)
        z = np.linspace(-0.3, 0.3, 4)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.array([X, Y, Z]).T.reshape(-1, 3)
        
        # Add jitter
        jitter = rng.normal(0, 0.02, points.shape)
        points += jitter
        # Clip to box
        points = np.clip(points, -0.49, 0.49)
        return points
    else:
        return rng.uniform(-0.4, 0.4, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Try different initialization strategies and seeds
    # We will run multiple optimizations. Each L-BFGS-B run takes some time.
    # For N=100, it should be fast (milliseconds to seconds).
    
    methods = ['grid_jitter', 'random']
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Shuffle seeds and methods to diversify
    rng_master = np.random.default_rng(42)
    trial_order = []
    for m in methods:
        for s in seeds:
            trial_order.append((m, s))
    rng_master.shuffle(trial_order)
    
    for method, seed in trial_order:
        if time.time() > deadline - 10.0: # Stop if close to deadline
            break
            
        initial_points = _generate_initial_points(method, seed)
        
        # Optimize
        try:
            result = minimize(
                _repulsion_energy, 
                initial_points.flatten(), 
                jac=_repulsion_grad,
                method='L-BFGS-B',
                bounds=BOUNDS,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            optimized_centers = result.x.reshape(N, 3)
            
            # Check validity
            if not np.all(np.isfinite(optimized_centers)):
                continue
                
            # Calculate radius
            radius = _compute_radius(optimized_centers)
            
            if radius > best_radius:
                best_radius = radius
                best_centers = optimized_centers.copy()
                
        except Exception as e:
            continue
            
    # If we didn't find any valid result (unlikely), fallback to random
    if best_centers is None:
        best_centers = np.random.uniform(-0.4, 0.4, (N, 3))
        best_radius = _compute_radius(best_centers)
        
    # Final polish: A few steps of local search or just return
    # The L-BFGS-B result should be quite good.
    # We can try a quick SA polish if time permits.
    
    if time.time() < deadline - 5.0:
        # Quick SA polish on the best result
        current_centers = best_centers.copy()
        current_radius = best_radius
        T = 0.005
        alpha = 0.99
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            i = np.random.randint(N)
            move = np.random.normal(0, T, 3)
            new_centers = current_centers.copy()
            new_centers[i] += move
            if np.any(np.abs(new_centers[i]) > 0.49):
                T *= alpha
                continue
            new_radius = _compute_radius(new_centers)
            delta = new_radius - current_radius
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_centers = new_centers
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = current_centers.copy()
            T *= alpha
            
    return best_centers

# EVOLVE-BLOCK-END
