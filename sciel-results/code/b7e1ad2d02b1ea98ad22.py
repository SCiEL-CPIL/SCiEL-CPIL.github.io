"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    centers = np.asarray(centers)
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting for speed with N=50
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)

def _objective(x):
    """Objective function to minimize (negative radius)."""
    centers = x.reshape(-1, 3)
    # Check if centers are within bounds (with small tolerance)
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 1e10  # Penalty for out-of-bounds
    
    r = _compute_radius(centers)
    return -r

def _init_fcc():
    """Generate 50 points from an FCC lattice inside the cube."""
    # Lattice constant a
    a = 0.43
    points = []
    # Generate points in a range larger than the cube to ensure coverage
    # FCC basis vectors: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Grid of unit cells
    # We need to cover [-0.5, 0.5]
    # So i, j, k should range from -2 to 2 maybe?
    # a * i + basis
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * b
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    
    points = np.array(points)
    
    if len(points) < N:
        # If not enough points, fallback to random or grid
        return _init_grid()
    
    # Randomly select N points
    indices = np.random.choice(len(points), N, replace=False)
    return points[indices]

def _init_bcc():
    """Generate 50 points from a BCC lattice inside the cube."""
    a = 0.43
    points = []
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ])
    
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * b
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    
    points = np.array(points)
    
    if len(points) < N:
        return _init_grid()
    
    indices = np.random.choice(len(points), N, replace=False)
    return points[indices]

def _init_grid():
    """Create a 5x5x2 grid initialization."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.array([-0.25, 0.25])
    
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return centers

def _init_random():
    """Generate 50 random points inside the cube."""
    return np.random.uniform(-0.45, 0.45, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = None
    best_score = -np.inf
    
    # Set random seed for reproducibility in testing, but vary for diversity
    # Using different seeds for different initializations
    
    initializations = [
        ("fcc", _init_fcc()),
        ("bcc", _init_bcc()),
        ("grid", _init_grid()),
        ("random1", _init_random()),
        ("random2", _init_random()),
        ("random3", _init_random()),
    ]
    
    for name, centers in initializations:
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            break
        
        x0 = centers.flatten()
        current_score = _compute_radius(centers)
        
        if best_x is None or current_score > best_score:
            best_x = x0.copy()
            best_score = current_score
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _objective, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if np.all(np.isfinite(result.x)):
                r_opt = -result.fun
                if r_opt > best_score:
                    best_score = r_opt
                    best_x = result.x.copy()
                    
        except Exception as e:
            pass
    
    # If we have time, do a final refinement on the best solution
    if best_x is not None and time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                _objective, best_x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100}
            )
            if np.all(np.isfinite(result.x)):
                r_opt = -result.fun
                if r_opt > best_score:
                    best_score = r_opt
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_x is not None:
        centers = best_x.reshape(-1, 3)
        # Ensure all points are strictly inside the cube
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Final check for validity
        if not np.all(np.isfinite(centers)):
            # Fallback to clean grid
            centers = _init_grid()
        
        return centers
    else:
        # Fallback to clean grid
        return _init_grid()

# EVOLVE-BLOCK-END
