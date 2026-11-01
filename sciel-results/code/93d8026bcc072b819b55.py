"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _get_distance_components(c):
    """Get all distance constraints (pairwise and boundary) as a flat array."""
    n = c.shape[0]
    # Pairwise distances / 2
    pair_dists = pdist(c, metric='euclidean') / 2.0
    # Boundary distances: 0.5 - |x| and 0.5 - |y|
    bound_dists = 0.5 - np.abs(c)
    
    # Concatenate
    all_dists = np.concatenate([pair_dists, bound_dists.ravel()])
    return all_dists


def _soft_min_objective(params, beta):
    """Minimize negative soft-min of distances."""
    c = np.clip(params.reshape(-2, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    dists = _get_distance_components(c)
    
    # Ensure no negative values (numerical safety)
    dists = np.maximum(dists, 1e-12)
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow, shift by min
    d_min = np.min(dists)
    shifted = dists - d_min
    
    # Optimization: if beta is very large, just return -min
    if beta > 10000:
        return -d_min
    
    # Log-sum-exp trick
    terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(terms))
    
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initialize_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_softmin(c0, max_time=25.0):
    """
    Local optimization using a smooth soft-min surrogate, then polish with L-BFGS-B.
    Returns (best_c, best_r).
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # Phase 1: Soft-min annealing to get close to the optimum
    # Start with a small beta (smooth) and increase to a large beta (sharp)
    beta_schedule = [10.0, 50.0, 200.0, 1000.0]
    
    current_c = c0.copy()
    for beta in beta_schedule:
        if time.time() > start_time + max_time - 10:
            break
        
        def obj(x):
            return _soft_min_objective(x, beta)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            c_new = np.clip(result.x.reshape(-2, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            # Always update current_c to the result of the optimization step,
            # even if it didn't improve the hard radius, as it might be a better
            # starting point for the next sharper beta.
            current_c = c_new
            
        except Exception:
            pass
    
    # Phase 2: High-precision polish using the exact (non-smooth) objective
    # This helps "snap" to the true local optimum of the min function.
    if time.time() < start_time + max_time - 2:
        def obj_exact(x):
            c = np.clip(x.reshape(-2, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            return -_compute_radius(c)
        
        try:
            # Run a tight optimization on the exact objective
            result2 = minimize(
                obj_exact,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            c_new = np.clip(result2.x.reshape(-2, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                
        except Exception:
            pass
            
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    best_c = None
    best_r = 0.0
    
    # 1. Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_r:
        best_r = r_hex
        best_c = c_hex.copy()
    
    # 2. Grid initialization
    c_grid = _initialize_grid()
    r_grid = _compute_radius(c_grid)
    if r_grid > best_r:
        best_r = r_grid
        best_c = c_grid.copy()
    
    # 3. Optimize the best initialization found so far using the soft-min strategy
    if time.time() < deadline - 30:
        c_opt, r_opt = _local_optimize_softmin(best_c, max_time=min(30.0, deadline - time.time() - 5))
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # 4. Try a few random restarts if time permits
    np.random.seed(42)
    for i in range(5):
        if time.time() > deadline - 20:
            break
        
        # Random initialization, but maybe jittered grid to avoid bad local minima
        c_rand = c_grid + np.random.uniform(-0.01, 0.01, size=(N, 2))
        c_rand = np.clip(c_rand, -0.5, 0.5)
        
        r_rand = _compute_radius(c_rand)
        if r_rand > best_r:
            best_r = r_rand
            best_c = c_rand.copy()
        
        # Optimize this random start
        if time.time() < deadline - 10:
            c_opt, r_opt = _local_optimize_softmin(c_rand, max_time=min(15.0, deadline - time.time() - 5))
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        
        # Also try optimizing a purely random start
        if time.time() < deadline - 5:
            c_pure_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
            c_opt, r_opt = _local_optimize_softmin(c_pure_rand, max_time=min(8.0, deadline - time.time() - 2))
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()

    # Final safety check
    if best_c is None:
        best_c = c_grid
    
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
