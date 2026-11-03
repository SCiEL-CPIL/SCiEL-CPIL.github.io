"""Structural change: Global optimization via differential evolution on exact min-clearance objective."""
N = 200
import numpy as np
import time
from scipy.optimize import differential_evolution

_DEADLINE = 145.0  # Safety margin below 150s kill
_START_TIME = time.time()


def _check_time():
    return time.time() - _START_TIME < _DEADLINE


def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5, 0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _objective(centers_flat):
    """
    Objective for differential_evolution: we want to maximize the minimum clearance.
    differential_evolution minimizes, so we return -min_clearance.
    """
    centers = centers_flat.reshape(-1, 2)
    # Clip to be safe, though DE bounds should handle this
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    # Boundary distances
    b_x = 0.5 - np.abs(centers[:, 0])
    b_y = 0.5 - np.abs(centers[:, 1])
    b_min = np.min(np.minimum(b_x, b_y))
    
    if b_min < 0:
        # Penalty if outside
        return 1.0 + abs(b_min)
        
    # Pairwise distances
    # For N=200, this is 40k distances. 
    # To speed up, we can early exit if we find a very small distance?
    # But we need the min.
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_d_sq = np.min(dists_sq)
    min_d = np.sqrt(min_d_sq)
    
    r = min(b_min, min_d / 2.0)
    return -r


def _initial_hex_grid(n_points):
    """Generate a hexagonal grid of points for initialization or reference."""
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    points = []
    row_height = s * np.sqrt(3) / 2
    n_rows = int(1.0 / row_height) + 2
    n_cols = int(1.0 / s) + 2
    
    total_width = (n_cols - 1) * s
    total_height = (n_rows - 1) * row_height
    
    start_x = -0.5 + (1.0 - total_width) / 2
    start_y = -0.5 + (1.0 - total_height) / 2
    
    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * s
            if i % 2 == 1:
                x += s / 2
            y = start_y + i * row_height
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > n_points:
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
        
    if len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
        
    return points[:n_points]


def _local_polish(centers, max_time=10.0):
    """
    Simple local polish using coordinate ascent or small random perturbations.
    We try to nudge points to increase the minimum clearance.
    """
    start_time = time.time()
    r_best = _compute_radius(centers)
    centers = centers.copy()
    
    # Identify tight pairs and boundaries
    # We will iterate: find the point with the smallest margin, try to move it.
    for _ in range(200):
        if time.time() - start_time > max_time:
            break
        if not _check_time():
            break
            
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_margins = dists / 2.0
        
        bound_x = 0.5 - np.abs(centers[:, 0])
        bound_y = 0.5 - np.abs(centers[:, 1])
        bound_margins = np.minimum(bound_x, bound_y)
        
        min_pair_per_point = np.min(pair_margins, axis=1)
        min_margin_per_point = np.minimum(min_pair_per_point, bound_margins)
        
        idx_min = np.argmin(min_margin_per_point)
        
        # Try moving this point in a few directions
        # 1. Away from nearest neighbor
        # 2. Away from boundary
        # 3. Random small steps
        
        moved = False
        
        # Direction 1: Away from nearest neighbor
        j = np.argmin(pair_margins[idx_min])
        vec = centers[idx_min] - centers[j]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
            for step in [0.005, 0.002, 0.001]:
                new_c = centers.copy()
                new_c[idx_min] += vec * step
                new_c[idx_min, 0] = np.clip(new_c[idx_min, 0], -0.5, 0.5)
                new_c[idx_min, 1] = np.clip(new_c[idx_min, 1], -0.5, 0.5)
                r_new = _compute_radius(new_c)
                if r_new > r_best:
                    centers = new_c
                    r_best = r_new
                    moved = True
                    break
        
        # Direction 2: Away from boundary
        if not moved:
            # Determine which boundary is closest
            if bound_x[idx_min] < bound_y[idx_min]:
                vec = np.array([-np.sign(centers[idx_min, 0]), 0])
            else:
                vec = np.array([0, -np.sign(centers[idx_min, 1])])
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
                for step in [0.005, 0.002, 0.001]:
                    new_c = centers.copy()
                    new_c[idx_min] += vec * step
                    new_c[idx_min, 0] = np.clip(new_c[idx_min, 0], -0.5, 0.5)
                    new_c[idx_min, 1] = np.clip(new_c[idx_min, 1], -0.5, 0.5)
                    r_new = _compute_radius(new_c)
                    if r_new > r_best:
                        centers = new_c
                        r_best = r_new
                        moved = True
                        break
                        
        # Direction 3: Random steps
        if not moved:
            for _ in range(10):
                step_vec = np.random.normal(0, 0.002, size=2)
                new_c = centers.copy()
                new_c[idx_min] += step_vec
                new_c[idx_min, 0] = np.clip(new_c[idx_min, 0], -0.5, 0.5)
                new_c[idx_min, 1] = np.clip(new_c[idx_min, 1], -0.5, 0.5)
                r_new = _compute_radius(new_c)
                if r_new > r_best:
                    centers = new_c
                    r_best = r_new
                    moved = True
                    break
                    
    return centers, r_best


def construct():
    start_time = time.time()
    
    # Define bounds for differential evolution
    # Centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Initial population
    # We can provide a seed or let DE generate. 
    # Providing a good initial population (hex grid) might help.
    # DE in SciPy takes `x0` as a list of initial points or a single point.
    # If we pass a list, it uses them as the initial population.
    # Let's generate a few diverse initial candidates.
    
    init_candidates = []
    
    # 1. Hex grid
    hex_grid = _initial_hex_grid(N)
    hex_grid += np.random.normal(0, 0.001, size=hex_grid.shape)
    hex_grid[:, 0] = np.clip(hex_grid[:, 0], -0.49, 0.49)
    hex_grid[:, 1] = np.clip(hex_grid[:, 1], -0.49, 0.49)
    init_candidates.append(hex_grid.flatten())
    
    # 2. Random uniform
    rand_grid = np.random.uniform(-0.48, 0.48, size=(N, 2))
    init_candidates.append(rand_grid.flatten())
    
    # 3. Another hex grid with different phase
    hex_grid2 = _initial_hex_grid(N)
    hex_grid2 += np.random.normal(0, 0.005, size=hex_grid2.shape)
    hex_grid2[:, 0] = np.clip(hex_grid2[:, 0], -0.49, 0.49)
    hex_grid2[:, 1] = np.clip(hex_grid2[:, 1], -0.49, 0.49)
    init_candidates.append(hex_grid2.flatten())
    
    # Run Differential Evolution
    # maxiter: number of iterations
    # popsize: size of the population
    # tol: tolerance
    # strategy: 'best1bin' is default
    
    # We need to be careful with time. 
    # Each evaluation is O(N^2) = 40k ops. 
    # With popsize=20, that's 800k ops per iteration.
    # 100 iterations = 80M ops. Should be fast enough in NumPy.
    
    try:
        res = differential_evolution(
            _objective,
            bounds,
            x0=init_candidates,
            maxiter=50,
            popsize=20,
            tol=1e-6,
            strategy='best1bin',
            seed=42
        )
        
        best_centers = res.x.reshape(-1, 2)
        best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5, 0.5)
        best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5, 0.5)
        
        r_best = _compute_radius(best_centers)
        
    except Exception as e:
        # Fallback to hex grid if DE fails
        best_centers = _initial_hex_grid(N)
        r_best = _compute_radius(best_centers)
        
    # Local Polish
    if _check_time() and (145.0 - (time.time() - _START_TIME)) > 5.0:
        best_centers, r_best = _local_polish(best_centers, max_time=10.0)
        
    # Final validation
    r_final = _compute_radius(best_centers)
    if r_final <= 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
