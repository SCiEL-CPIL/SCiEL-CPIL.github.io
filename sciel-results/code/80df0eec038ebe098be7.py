"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


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


def _soft_min_objective(params, beta=50.0):
    """
    Smooth surrogate for the negative radius using soft-min (log-sum-exp).
    We want to maximize min(pairwise_dists/2, boundary_dists).
    Soft-min approximates the minimum: -soft_min = soft_max of negatives.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean')
    pair_vals = dists / 2.0
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Combine all constraints into one array
    all_vals = np.concatenate([pair_vals, boundary_dists.flatten()])
    
    # Soft-min: -log-sum-exp(-beta * x) / beta approximates min(x) for large beta
    # We want to maximize the soft-min, so minimize negative soft-min
    max_val = np.max(all_vals)
    # Numerically stable log-sum-exp
    shifted = -beta * (all_vals - max_val)
    lse = np.log(np.sum(np.exp(shifted))) + beta * max_val
    soft_min_val = lse / beta
    
    return -soft_min_val


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.06, 0.08, 15):
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
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _force_directed_init(c0, n_iter=500, k=0.005):
    """
    Force-directed relaxation: circles repel each other and are attracted to center.
    This helps untangle overlaps from random initialization.
    """
    c = c0.copy()
    n = c.shape[0]
    
    for it in range(n_iter):
        # Compute repulsive forces between all pairs
        # For efficiency, only consider pairs closer than some threshold
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion: F = k / dist^2 * direction (capped to avoid blowup)
        min_dist = 0.01
        safe_dists = np.maximum(dists, min_dist)
        forces = np.zeros_like(c)
        
        # Only compute for close pairs to save time
        mask = safe_dists < 0.15
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = safe_dists[i, j]
                    f = k / (d * d)
                    forces[i] += f * (c[i] - c[j]) / d
                    forces[j] -= f * (c[i] - c[j]) / d
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Damping
        damping = 0.9 * (1.0 - it / n_iter) + 0.1
        c += damping * forces
        
        # Clip to bounds
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return c


def _local_optimize_smooth(c0, max_time=25.0, beta_start=20.0, beta_end=200.0):
    """
    Local optimization using L-BFGS-B with a smooth surrogate objective.
    Uses annealing of beta to gradually sharpen the soft-min toward the true min.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: Smooth optimization with increasing beta
    betas = [beta_start, 50.0, 100.0, beta_end]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        def _obj(params):
            return _soft_min_objective(params, beta=beta)
        
        try:
            result = minimize(
                _obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Phase 2: High-precision polish with exact objective
    if time.time() < start_time + max_time - 2:
        def _exact_obj(params):
            c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r = _compute_radius(c)
            return -r
        
        try:
            result = minimize(
                _exact_obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with smooth surrogate."""
    return _local_optimize_smooth(c0, max_time=max_time)


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # Initialize with hexagonal pattern
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Optimize hexagonal initialization
    if time.time() < deadline - 10:
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, deadline - time.time() - 5))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try force-directed initialization from random starts
    np.random.seed(123)
    for trial in range(4):
        if time.time() > deadline - 25:
            break
        
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        # Apply force-directed relaxation
        c_fd = _force_directed_init(c_rand, n_iter=300, k=0.005)
        r_fd = _compute_radius(c_fd)
        
        if r_fd > best_radius:
            best_radius = r_fd
            best_c = c_fd.copy()
        
        if time.time() < deadline - 20:
            c_opt, r_opt = _local_optimize_smooth(c_fd, max_time=min(25.0, deadline - time.time() - 5))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Final polish pass on the best solution found
    if best_c is not None and time.time() < deadline - 5:
        c_polish, r_polish = _local_optimize_smooth(best_c, max_time=min(10.0, deadline - time.time() - 2))
        if r_polish > best_radius:
            best_radius = r_polish
            best_c = c_polish.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
