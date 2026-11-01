"""Fifty equal circles in a unit square - local optimization with multi-start."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(points):
    """Compute the maximum common radius for given circle centres."""
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are within [-0.5, 0.5]^2
    if np.any(np.abs(points) > 0.5 + 1e-12):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Distance to walls
    min_x = 0.5 - np.max(points[:, 0])
    max_x = 0.5 + np.min(points[:, 0])
    min_y = 0.5 - np.max(points[:, 1])
    max_y = 0.5 + np.min(points[:, 1])
    min_wall = min(min_x, max_x, min_y, max_y)
    return min(min_pair / 2.0, min_wall)

def _neg_radius(points_flat):
    """Objective function: negative of the radius (to minimize)."""
    points = points_flat.reshape(N, 2)
    r = _compute_radius(points)
    return -r

def _hexagonal_init(rng):
    """Generate initial configuration with hexagonal-like arrangement."""
    # Start with a grid and add small perturbations
    m = 7
    n = 7
    # Create a 7x7 grid and pick 50 points
    xs = np.linspace(-0.4, 0.4, m)
    ys = np.linspace(-0.4, 0.4, n)
    grid = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # Offset every other column for hexagonal packing
            offset = 0.0 if i % 2 == 0 else (ys[1] - ys[0]) / 2
            grid.append([x, y + offset])
    grid = np.array(grid[:N])
    # Add small random perturbation
    grid += rng.uniform(-0.01, 0.01, size=grid.shape)
    # Clip to stay within bounds
    grid = np.clip(grid, -0.45, 0.45)
    return grid

def _grid_init(rng):
    """Generate initial configuration with grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    grid += rng.uniform(-0.005, 0.005, size=grid.shape)
    grid = np.clip(grid, -0.45, 0.45)
    return grid

def _random_init(rng):
    """Generate random initial configuration."""
    points = rng.uniform(-0.45, 0.45, size=(N, 2))
    return points

def _local_optimize(points, deadline, rng):
    """Run local optimization from given initial points."""
    best_points = points.copy()
    best_radius = _compute_radius(best_points)
    
    flat = best_points.flatten()
    
    # Try multiple optimization attempts with different parameters
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                _neg_radius,
                flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            continue
        
        if result.success or result.fun < -best_radius:
            new_points = result.x.reshape(N, 2)
            new_radius = _compute_radius(new_points)
            if new_radius > best_radius:
                best_radius = new_radius
                best_points = new_points
                flat = result.x
    
    return best_points, best_radius

def construct():
    """Construct the arrangement of 50 circles."""
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = 0.0
    
    # Multiple initializations
    inits = []
    
    # Grid initialization
    inits.append(_grid_init(rng))
    
    # Hexagonal initialization
    inits.append(_hexagonal_init(rng))
    
    # Several random initializations
    for _ in range(5):
        inits.append(_random_init(rng))
    
    # Try each initialization
    for init_points in inits:
        if time.time() > deadline:
            break
        
        points, radius = _local_optimize(init_points, deadline, rng)
        
        if radius > best_radius:
            best_radius = radius
            best_points = points
    
    # Final polish: more iterations from best point
    if best_points is not None and time.time() < deadline - 5:
        flat = best_points.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        try:
            result = minimize(
                _neg_radius,
                flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.success:
                new_points = result.x.reshape(N, 2)
                new_radius = _compute_radius(new_points)
                if new_radius > best_radius:
                    best_points = new_points
                    best_radius = new_radius
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_points = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Ensure valid output
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
