import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    # Boundary distances: distance from each centre to the nearest wall
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    # Pairwise distances
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_objective(params, beta):
    """Minimize negative soft-min of all distance constraints."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Get all distance components
    pair_dists = pdist(c, metric='euclidean') / 2.0
    bound_dists = 0.5 - np.abs(c)
    all_dists = np.concatenate([pair_dists, bound_dists.ravel()])
    
    # Ensure no negative values
    all_dists = np.maximum(all_dists, 1e-12)
    
    if beta > 10000:
        return -np.min(all_dists)
    
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(terms))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.085, 20):
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
    
    # Fallback: square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _initialize_grid():
    """Initialize with a rectangular grid."""
    # Try to find best w, h such that w*h >= N and aspect ratio close to 1
    best_c = None
    best_r = 0
    
    for w in range(12, 18):
        h = int(np.ceil(N / w))
        if w * h < N:
            continue
        
        ticks_x = (np.arange(w) + 0.5) / w - 0.5
        ticks_y = (np.arange(h) + 0.5) / h - 0.5
        grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
        
        if len(grid) >= N:
            c = grid[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    return _initialize_hexagonal()

def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with soft-min objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    from scipy.optimize import minimize
    
    # First pass: moderate beta
    betas = [50, 200, 1000]
    
    for beta in betas:
        if time.time() - start_time > max_time - 3:
            break
        
        try:
            def obj(params):
                return _soft_min_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 0:
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
        except Exception:
            pass
    
    # Final polish with very high beta (near-exact)
    if time.time() - start_time < max_time - 2:
        try:
            beta = 5000
            def obj(params):
                return _soft_min_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < 0:
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # 1. Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Grid initialization
    c_grid = _initialize_grid()
    r_grid = _compute_radius(c_grid)
    if r_grid > best_radius:
        best_radius = r_grid
        best_c = c_grid.copy()
    
    # 3. Local optimization on best initial
    if time.time() < deadline - 10:
        c_opt, r_opt = _local_optimize(best_c, max_time=min(30.0, deadline - time.time() - 5))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # 4. Try a few random restarts with optimization
    rng = np.random.default_rng(123)
    for trial in range(3):
        if time.time() > deadline - 25:
            break
        
        # Random initialization with some structure
        c_rand = rng.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()
        
        if time.time() < deadline - 15:
            c_opt, r_opt = _local_optimize(c_rand, max_time=min(20.0, deadline - time.time() - 5))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # 5. Final polish on the best solution
    if time.time() < deadline - 5:
        c_final, r_final = _local_optimize(best_c, max_time=min(10.0, deadline - time.time() - 2))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
