import numpy as np
import scipy.optimize as opt
import time

N = 50

def _compute_radius(centers):
    """
    Computes the exact maximum radius for the given centers.
    Returns 0.0 if invalid (out of bounds or coincident).
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    min_pair_dist_sq = np.min(dists_sq[mask])
    
    if min_pair_dist_sq < 1e-12:
        return 0.0
        
    min_pair_dist = np.sqrt(min_pair_dist_sq)
    
    # Boundary distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    min_wall_dist = np.min(np.minimum(dx, dy))
    
    return np.min([min_pair_dist / 2.0, min_wall_dist])

def _soft_min(values, alpha):
    """
    Smooth approximation of min(values).
    values: 1D numpy array of positive floats.
    """
    if len(values) == 0:
        return 0.0
    v = np.asarray(values)
    v_min = np.min(v)
    shifted = v - v_min
    exponents = -alpha * shifted
    # Clip to avoid overflow in exp for very negative exponents (shouldn't happen since shifted >= 0)
    # But if alpha is huge and shifted is tiny, it's fine.
    # np.exp(-large) -> 0
    # np.exp(0) -> 1
    # We want -log(sum(exp(-alpha * shifted))) / alpha
    # Using logsumexp for stability
    # log(sum(exp(x))) = logsumexp(x)
    # Here x = -alpha * shifted
    # logsumexp(-alpha * shifted)
    # We can use scipy.special.logsumexp or implement manually
    # Manual:
    max_exp = np.max(exponents)
    if max_exp > 700: # overflow protection, though shifted >= 0 so exponents <= 0
         return v_min
    sum_exp = np.sum(np.exp(exponents - max_exp))
    log_sum = max_exp + np.log(sum_exp)
    return v_min - log_sum / alpha

def _objective_smooth(centers_flat, alpha):
    centers = centers_flat.reshape(-1, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    pair_dists = np.sqrt(dists_sq[mask]) / 2.0
    
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)
    
    # Combine
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # We want to maximize the soft-min, so minimize negative soft-min
    return -_soft_min(all_dists, alpha)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    np.random.seed(42)
    
    best_radius = 0.0
    best_centers = None
    
    # Initial configurations
    inits = []
    
    # 1. 7x7 grid + 1 center
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init1 = grid[:49]
    init1 = np.vstack([init1, [[0.0, 0.0]]])
    inits.append(init1.copy())
    
    # 2. 8x8 grid subset
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid8 = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid8[:50].copy())
    
    # 3. Hexagonal lattice initialization (jittered)
    # Create a hexagonal lattice of points and select 50
    # Spacing s. In a unit square, we can fit about 1/s points per side.
    # For r ~ 0.07, s ~ 2r ~ 0.14.
    # 1/0.14 ~ 7 points.
    # Let's generate a larger hex grid and pick the best 50 or just a subset.
    # Or simply generate a grid with hexagonal offsets.
    # Row y: x offsets alternate.
    hex_points = []
    s = 0.15 # spacing
    y = -0.5 + s/2
    row = 0
    while y <= 0.5:
        x_offset = (s * np.sqrt(3) / 2) * (row % 2)
        x = -0.5 + s/2 + x_offset
        while x <= 0.5:
            hex_points.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    
    if len(hex_points) >= N:
        # Add some jitter
        hex_init = np.array(hex_points[:N]) + np.random.uniform(-0.01, 0.01, (N, 2))
        # Clip to bounds
        hex_init = np.clip(hex_init, -0.49, 0.49)
        inits.append(hex_init)
    
    # 4. Random inits
    for _ in range(15):
        rand = np.random.uniform(-0.45, 0.45, (N, 2))
        inits.append(rand)
        
    # Optimization parameters
    alphas = [10, 50, 100, 200, 500, 1000, 2000]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current = init.copy()
        
        # Annealing-like sequence of alphas
        for alpha in alphas:
            if time.time() > deadline:
                break
                
            try:
                res = opt.minimize(
                    _objective_smooth,
                    current.flatten(),
                    args=(alpha,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
                
        r = _compute_radius(current)
        if r > best_radius:
            best_radius = r
            best_centers = current.copy()
            
        # If we have time, do a final high-precision polish on the best so far
        # Actually, let's do a polish on each candidate before moving to next?
        # Or just keep the best.
        
        # Add a small perturbation to the current best to escape local minima?
        # The random inits cover that.
        
    if best_centers is None:
        # Fallback
        return inits[0]
        
    return best_centers
