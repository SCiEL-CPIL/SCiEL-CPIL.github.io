"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the largest common radius for given centers in [-0.5, 0.5]^3."""
    # Wall distance: min over each center and each axis of (0.5 - |c_ik|)
    abs_c = np.abs(centers)
    wall_dists = 0.5 - abs_c
    wall_min = np.min(wall_dists)
    
    # Pairwise distances
    n = len(centers)
    # Use broadcasting for pairwise distances (N=50 is small)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists) / 2.0
    
    return min(wall_min, pair_min)

def _lloyd_step(centers, cube_half=0.5, max_iter=200, tol=1e-10):
    """
    Lloyd's algorithm for Voronoi-based sphere packing.
    Move each center to the centroid of its Voronoi cell clipped to the cube.
    """
    centers = centers.copy().astype(float)
    n = len(centers)
    
    # We work in the cube [-0.5, 0.5]^3
    # For Voronoi, we need to handle boundary cells properly.
    # Strategy: add "ghost" points outside the cube to clip Voronoi cells,
    # or use a different approach.
    
    # Better approach: Use the fact that for points inside a convex polytope,
    # the Voronoi cell of point i is the intersection of half-spaces.
    # For the cube, we can compute the centroid of the clipped cell.
    
    # Simpler approach: use scipy's Voronoi but it doesn't handle boundary well.
    # Instead, let's use a gradient-based approach with a smooth objective.
    
    # Actually, let's implement a proper Lloyd step using the following:
    # For each point, compute its Voronoi cell within the cube, then move to centroid.
    # This is complex. Let's use a simpler but effective approach:
    # Iteratively push apart close pairs and pull toward center of empty space.
    
    # Let's use a different strategy: simulated annealing / local search
    # with a smooth objective.
    
    return centers

def _smooth_min(values, beta=100.0):
    """Soft-min approximation using log-sum-exp."""
    values = np.asarray(values, dtype=float)
    return -1.0 / beta * np.log(np.sum(np.exp(-beta * values)))

def _objective(x_flat, beta=50.0):
    """
    Objective to minimize: negative of the minimum radius.
    We use a soft-min to make it smooth.
    """
    centers = x_flat.reshape(-1, 3)
    
    # Wall distances
    abs_c = np.abs(centers)
    wall_dists = 0.5 - abs_c
    wall_min = np.min(wall_dists)
    
    # Pairwise distances / 2
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraints into a single soft-min
    # We want to maximize min(wall_min, pair_min)
    # Equivalently, minimize -min(...)
    
    # Use soft-min over all pairwise distances and wall distances
    all_constraints = np.concatenate([np.min(wall_dists, axis=1), pair_dists[np.triu_indices(n, k=1)]])
    
    # Soft-min of all constraints
    soft_min_val = -1.0 / beta * np.log(np.sum(np.exp(-beta * all_constraints)))
    
    # We want to maximize this, so minimize its negative
    return -soft_min_val

def _local_search(centers, max_iter=500, lr=0.01, beta_schedule=None):
    """
    Gradient-based local search with adaptive learning rate.
    """
    centers = centers.copy().astype(float)
    n = len(centers)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    # Start with a reasonable beta
    beta = 30.0
    
    for it in range(max_iter):
        x0 = centers.flatten()
        
        # Use a wrapper that adds a small penalty for being too close to walls
        def obj(x):
            c = x.reshape(-1, 3)
            # Clip to stay in bounds
            c = np.clip(c, -0.4999, 0.4999)
            return _objective(c.flatten(), beta=beta)
        
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        
        new_centers = res.x.reshape(-1, 3)
        new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            # Increase beta to sharpen the soft-min
            beta *= 1.2
        else:
            # No improvement, increase beta and try again
            beta *= 1.1
            if beta > 500:
                break
    
    return best_centers, best_radius

def _random_init(n=50, seed=42):
    """Generate random initial positions in the cube."""
    rng = np.random.RandomState(seed)
    # Start with a grid-like structure for better initial spread
    m = int(np.ceil(n ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= n:
        return grid[:n].astype(float)
    else:
        # Fill remaining with random
        remaining = n - len(grid)
        rand = rng.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([grid, rand])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple restarts with different seeds
    seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Random initialization
        centers = _random_init(N, seed=seed)
        
        # Add small random perturbation to break symmetry
        rng = np.random.RandomState(seed)
        centers = centers + rng.normal(0, 0.02, size=centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        
        # Local search
        t0 = time.time()
        centers, radius = _local_search(centers, max_iter=200)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final polish with higher precision
    if best_centers is not None and time.time() < deadline - 10:
        best_centers, best_radius = _local_search(best_centers, max_iter=300, beta_schedule=None)
    
    # Ensure valid output
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_radius = _compute_radius(best_centers)
    
    # Final validation and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
