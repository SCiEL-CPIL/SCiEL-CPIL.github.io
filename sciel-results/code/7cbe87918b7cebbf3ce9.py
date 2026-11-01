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
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta=200.0):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    # For numerical stability, shift by min
    shifted = values - v_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta=200.0):
    """
    Smooth objective using soft-min for the radius.
    Radius r = min(min_pair_dist/2, min_boundary)
    We minimize -r, so we want to maximize r.
    Using soft-min on the negative of the constraints isn't quite right for min.
    r = min(A, B) where A = min_pair_dist/2, B = min_boundary.
    Actually, r = min( min_{i<j} d_ij/2, min_i b_i ).
    Let's define the set of "distances" D = { d_ij/2 for all i<j } U { b_i for all i }.
    r = min(D).
    To maximize r, we minimize -min(D) = max(-D).
    So we want to minimize the soft-max of -D.
    soft_max(x, beta) = (1/beta) * log(sum(exp(beta*x)))
    So objective = soft_max(-D, beta).
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Pairwise distances / 2
    if n > 1:
        dists = pdist(c, metric='euclidean') / 2.0
    else:
        dists = np.array([])
    
    # Combine all "radius candidates"
    # We want min of all these.
    # So we want to minimize -min(all) = max(-all).
    all_candidates = np.concatenate([dists, boundary_dists])
    
    # Soft max of -all_candidates
    neg_candidates = -all_candidates
    # Shift for stability
    max_neg = np.max(neg_candidates)
    if max_neg == -np.inf:
        return 0.0
    shifted = neg_candidates - max_neg
    # exp(beta * shifted) might overflow if beta is large and shifted is positive?
    # shifted <= 0, so exp(beta * shifted) <= 1. No overflow.
    sum_exp = np.sum(np.exp(beta * shifted))
    soft_max_val = max_neg + np.log(sum_exp) / beta
    
    return soft_max_val


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    # Wider range and finer steps
    for d in np.linspace(0.05, 0.09, 30):
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
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B on the smooth objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # Use a moderate beta initially for smoothness, then increase
    # Start with beta=100
    try:
        result = minimize(
            _smooth_objective,
            c0.flatten(),
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < _smooth_objective(best_c.flatten(), 100.0):
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
    except Exception:
        pass
    
    # Second pass with higher beta (sharper) if time allows
    if time.time() - start_time < max_time - 10:
        try:
            result2 = minimize(
                _smooth_objective,
                best_c.flatten(),
                args=(500.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
            
    # Final high-precision polish on the exact (nonsmooth) objective?
    # L-BFGS-B struggles with nonsmoothness. But maybe a few iterations help?
    # Let's try one more pass with the exact objective if time permits, with very tight tol.
    if time.time() - start_time < max_time - 5:
        def _exact_obj(params):
            c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r = _compute_radius(c)
            return -r
        
        try:
            result3 = minimize(
                _exact_obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            c_new = np.clip(result3.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass

    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # 1. Hexagonal Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # 2. Optimize from Hexagonal
    if time.time() < deadline - 20:
        c_opt, r_opt = _local_optimize(c_hex, max_time=min(40.0, deadline - time.time() - 10))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
            
    # 3. Perturbed Restarts
    np.random.seed(123)
    for trial in range(3):
        if time.time() > deadline - 30:
            break
        
        # Perturb the current best
        sigma = 0.01
        c_pert = best_c + np.random.normal(0, sigma, size=best_c.shape)
        c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
        
        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
            
        if time.time() < deadline - 25:
            c_opt, r_opt = _local_optimize(c_pert, max_time=min(20.0, deadline - time.time() - 15))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
                
    # 4. Random Initialization Fallback
    if time.time() < deadline - 35:
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()
            
        if time.time() < deadline - 20:
            c_opt, r_opt = _local_optimize(c_rand, max_time=min(15.0, deadline - time.time() - 10))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()

    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
