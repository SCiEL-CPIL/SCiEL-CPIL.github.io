import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0

def _compute_radius(c):
    """
    Compute the maximum radius for a set of centers c of shape (N, 3).
    Returns 0.0 if invalid.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    
    # Pairwise distances
    # c[:, np.newaxis, :] - c[np.newaxis, :, :] -> (N, N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # c is in [-0.5, 0.5]
    # dist to x=-0.5 is c[:,0] + 0.5
    # dist to x=0.5 is 0.5 - c[:,0]
    # min of these for each sphere, then min over all spheres
    wall_dists = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius

def _soft_min_objective(params, beta=100.0):
    """
    Objective to minimize: -soft_min(constraints).
    params: flattened array of shape (N*3,)
    """
    c = params.reshape(N, 3)
    
    # Penalty for out of bounds to help optimizer stay in range
    # If any point is out, return a large positive number
    if np.any(c < -0.5) or np.any(c > 0.5):
        # Calculate how far out to scale penalty
        out = np.maximum(0, np.maximum(c - 0.5, -0.5 - c))
        return 10.0 + np.sum(out**2)

    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We want constraints: d_ij >= 2r  => d_ij/2 >= r
    # and wall_dist >= r
    # So we take soft min of (dists/2) and (wall_dists)
    
    # Extract upper triangle for pairwise to avoid double counting and inf
    iu = np.triu_indices(N, 1)
    pair_vals = dists[iu] / 2.0
    
    wall_dists = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
    
    all_vals = np.concatenate([pair_vals, wall_dists])
    
    # Soft min
    # S = -1/beta * log(sum(exp(-beta * v)))
    # Shift for stability
    min_val = np.min(all_vals)
    exps = np.exp(-beta * (all_vals - min_val))
    log_sum = np.log(np.sum(exps))
    soft_min_val = min_val - log_sum / beta
    
    return -soft_min_val

def _halton_init(n, base=(2, 3, 5)):
    """Generate Halton sequence points in [0,1]^3, scale to [-0.5, 0.5]."""
    def halton(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        for j in range(3):
            points[i-1, j] = halton(i, base[j])
            
    # Scale to [-0.5, 0.5]
    points = points * 1.0 - 0.5
    return points

def _bcc_init(n):
    """
    Generate BCC lattice points.
    BCC has 2 points per unit cell.
    We need to fit ~100 points.
    Let grid be g x g x g.
    Points per cell = 2.
    Total = 2 * g^3.
    For g=4, 2*64=128.
    For g=3, 2*27=54.
    So g=4 gives 128 points. We can select 100 best or just take first 100?
    Better to generate a grid of 128 and pick the 100 with largest clearance?
    Or just use the first 100.
    Let's generate a 4x4x4 BCC grid.
    Unit cell size = 1/4 = 0.25?
    If we have 4 cells per side, total length 1.
    Cell size a = 0.25.
    Corners: (0,0,0), (0.25,0,0)...
    Centers: (0.125, 0.125, 0.125)...
    Coordinates in [-0.5, 0.5]:
    Shift by -0.5.
    """
    g = 4
    a = 1.0 / g
    points = []
    for i in range(g):
        for j in range(g):
            for k in range(g):
                # Corner
                x, y, z = i*a, j*a, k*a
                points.append([x-0.5, y-0.5, z-0.5])
                # Body center
                points.append([x+a/2-0.5, y+a/2-0.5, z+a/2-0.5])
                
    points = np.array(points)
    # We have 128 points. We need 100.
    # Calculate radius for all 128? No, we need to select 100.
    # Simple heuristic: keep the ones that are "most central" or just random subset?
    # Actually, a better init is to use a 5x5x4 grid of simple cubic, or just use Halton.
    # BCC init is tricky to subset. Let's just use Halton and Random.
    # Or, we can use a 4x4x4 BCC and randomly remove 28 points to see if it improves?
    # For now, let's just return the first 100 for simplicity, or better, 
    # let's construct a specific 100-point BCC-like structure.
    # Actually, 100 is 2 * 50. Not a perfect cube.
    # Let's stick to Halton and Random for init diversity.
    
    # If we must use BCC, we can take a 5x5x4 simple grid and perturb?
    # Let's just return Halton for this function to keep it simple, or implement a proper subset selection.
    # For now, return a random subset of the 128 points?
    idx = np.random.choice(len(points), 100, replace=False)
    return points[idx]

def _random_init():
    return np.random.uniform(-0.5, 0.5, (N, 3))

def _anneal(c0, max_time=60.0):
    """
    Simulated Annealing to improve the packing.
    """
    start_time = time.time()
    c = c0.copy()
    r_curr = _compute_radius(c)
    r_best = r_curr
    c_best = c.copy()
    
    T = 0.05  # Initial temperature
    T_min = 1e-5
    alpha = 0.999  # Cooling rate
    step_size = 0.05
    
    while time.time() - start_time < max_time:
        # Pick a random sphere
        i = np.random.randint(N)
        # Generate a random move
        move = np.random.normal(0, step_size, 3)
        
        c_new = c.copy()
        c_new[i] += move
        # Keep in bounds? SA can handle out of bounds with penalty, 
        # but _compute_radius returns 0 if out of bounds. 
        # It's better to project back to bounds or use a penalty.
        # Let's just clip to bounds for now, or allow out of bounds and let radius be 0.
        # If radius is 0, delta is large negative, so it will be rejected unless T is high.
        # This might cause many rejections. 
        # Better to move within bounds?
        # Let's just use the raw move and let the radius function handle it.
        
        r_new = _compute_radius(c_new)
        delta = r_new - r_curr
        
        if delta > 0:
            # Improvement
            c = c_new
            r_curr = r_new
            if r_new > r_best:
                r_best = r_new
                c_best = c.copy()
        else:
            # Accept with probability exp(delta / T)
            # Note: delta is negative.
            if np.random.rand() < np.exp(delta / T):
                c = c_new
                r_curr = r_new
        
        T *= alpha
        if T < T_min:
            break
            
    return c_best, r_best

def _polish(c0, max_time=10.0):
    """
    Local optimization using L-BFGS-B.
    """
    start_time = time.time()
    params0 = c0.flatten()
    
    def obj(p):
        return _soft_min_objective(p, beta=100.0)
    
    try:
        res = minimize(
            obj,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxls': 20, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8},
            bounds=[(-0.5, 0.5)] * (N * 3)
        )
        c_res = res.x.reshape(N, 3)
        r_res = _compute_radius(c_res)
        return c_res, r_res
    except Exception:
        return c0, _compute_radius(c0)

def construct():
    start_time = time.time()
    
    # Initializations
    inits = []
    inits.append(_halton_init(N))
    inits.append(_random_init())
    # Add a few random inits
    for _ in range(2):
        inits.append(_random_init())
    
    best_c = None
    best_r = 0.0
    
    # Evaluate initializations
    for c in inits:
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
    # Simulated Annealing
    # We have 115s limit.
    # Let's spend 80s on SA and 30s on polishing multiple candidates.
    sa_time = 80.0
    
    # Run SA on the best init
    c_sa, r_sa = _anneal(best_c, max_time=sa_time)
    if r_sa > best_r:
        best_r = r_sa
        best_c = c_sa.copy()
    
    # Polish the result
    if time.time() - start_time < DEADLINE - 5:
        c_polished, r_polished = _polish(best_c, max_time=20.0)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
            
    # If time remains, try SA on second best init?
    # For now, return the best found.
    
    return best_c

# EVOLVE-BLOCK-END
