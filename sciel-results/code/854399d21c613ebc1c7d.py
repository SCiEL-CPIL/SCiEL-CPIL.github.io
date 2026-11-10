import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(centers):
    """
    Compute the maximum common radius for a set of circle centers.
    Returns 0 if any center is outside or if there are coincident centers (dist=0).
    """
    # Check bounds
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    
    # Distance to boundary
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    dists = cdist(centers, centers)
    # Only look at upper triangle
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu[0], iu[1]].min() / 2.0
    
    return min(min_boundary, min_pair)

def _soft_min_radius(centers, beta=50.0):
    """
    Smooth approximation of min radius using log-sum-exp.
    This is differentiable and suitable for L-BFGS-B.
    """
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Compute all pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    half_dists = dists / 2.0
    min_d = half_dists.min()
    
    # Soft min: -log(sum(exp(-beta*x)))/beta
    # Use log-sum-exp trick for stability
    # sum(exp(-beta * (x - min_d)))
    exp_vals = np.exp(-beta * (half_dists - min_d))
    soft_min_dist = min_d - np.log(exp_vals.sum()) / beta
    
    return min(min_boundary, soft_min_dist)

def _hex_init(n):
    """Create a hexagonal grid initialization."""
    best_centers = None
    best_score = -1
    
    # Try different row/col configurations
    for rows in range(4, 14):
        cols = int(np.ceil(n / rows))
        if cols < 3 or cols > 15:
            continue
        
        dx = 1.0 / (cols + 0.5)
        dy = dx * np.sqrt(3) / 2
        
        x_start = -0.5 + dx / 2
        y_start = -0.5 + dy / 2
        
        centers = []
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                x = x_start + j * dx
                y = y_start + i * dy
                if i % 2 == 1:
                    x += dx / 2
                centers.append([x, y])
        
        if len(centers) >= n:
            centers = np.array(centers[:n])
            centers = np.clip(centers, -0.5, 0.5)
            s = _compute_radius(centers)
            if s > best_score:
                best_score = s
                best_centers = centers
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers

def _optimize_smooth(centers, time_budget):
    """
    Optimize using L-BFGS-B on the soft-min radius.
    """
    start_time = time.time()
    
    def objective(x_flat):
        if time.time() - start_time > time_budget:
            return 0.0
        centers_2d = x_flat.reshape(-1, 2)
        # Maximize soft radius = minimize negative soft radius
        r = _soft_min_radius(centers_2d, beta=100.0)
        return -r
    
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 300,
            'ftol': 1e-14,
            'gtol': 1e-8,
            'maxls': 50
        }
    )
    
    polished = result.x.reshape(-1, 2)
    polished = np.clip(polished, -0.5, 0.5)
    return polished

def _optimize_exact(centers, time_budget):
    """
    Optimize using Nelder-Mead on the exact (nonsmooth) radius.
    """
    start_time = time.time()
    
    def objective(x_flat):
        if time.time() - start_time > time_budget:
            return 0.0
        centers_2d = x_flat.reshape(-1, 2)
        # Maximize exact radius = minimize negative exact radius
        r = _compute_radius(centers_2d)
        return -r
    
    x0 = centers.flatten()
    
    # Nelder-Mead doesn't support bounds directly, but we can clip in the objective
    # or rely on the initialization being close to valid.
    # To be safe, we add a penalty if out of bounds.
    def penalized_objective(x_flat):
        if time.time() - start_time > time_budget:
            return 0.0
        centers_2d = x_flat.reshape(-1, 2)
        if np.any(np.abs(centers_2d) > 0.5):
            return 1000.0 # Large penalty
        r = _compute_radius(centers_2d)
        return -r
        
    result = minimize(
        penalized_objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-12
        }
    )
    
    polished = result.x.reshape(-1, 2)
    polished = np.clip(polished, -0.5, 0.5)
    return polished

def construct():
    start_time = time.time()
    total_time_limit = 140.0  # Leave 10s margin
    
    np.random.seed(42)
    
    best_centers = None
    best_score = -1
    
    # 1. Initialization
    centers = _hex_init(N)
    score = _compute_radius(centers)
    if score > best_score:
        best_centers = centers.copy()
        best_score = score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 2. Smooth Optimization (L-BFGS-B)
    # This helps escape the grid structure and find a better basin.
    if remaining > 60:
        smooth_result = _optimize_smooth(best_centers, time_budget=min(30, remaining - 5))
        smooth_score = _compute_radius(smooth_result)
        if smooth_score > best_score:
            best_centers = smooth_result
            best_score = smooth_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 3. Exact Optimization (Nelder-Mead)
    # This polishes the solution to the exact max-min value.
    if remaining > 30:
        exact_result = _optimize_exact(best_centers, time_budget=min(20, remaining - 5))
        exact_score = _compute_radius(exact_result)
        if exact_score > best_score:
            best_centers = exact_result
            best_score = exact_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # 4. Multi-Restart Local Search
    # Try a few perturbations and re-optimize to escape local minima.
    for restart in range(5):
        if time.time() - start_time > total_time_limit - 10:
            break
        
        # Perturb best centers
        perturbation_scale = 0.005 / (restart + 1)
        perturbed = best_centers + np.random.randn(N, 2) * perturbation_scale
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Quick smooth optimization
        if remaining > 15:
            smooth_res = _optimize_smooth(perturbed, time_budget=min(5, remaining - 5))
            s_score = _compute_radius(smooth_res)
            if s_score > best_score:
                best_centers = smooth_res
                best_score = s_score
        
        # Quick exact optimization
        if remaining > 10:
            exact_res = _optimize_exact(best_centers, time_budget=min(5, remaining - 5))
            e_score = _compute_radius(exact_res)
            if e_score > best_score:
                best_centers = exact_res
                best_score = e_score
        
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validation check
    final_r = _compute_radius(best_centers)
    if final_r <= 0 or not np.all(np.isfinite(best_centers)):
        best_centers = _hex_init(N)
    
    return best_centers
