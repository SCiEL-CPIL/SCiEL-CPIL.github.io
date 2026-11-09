# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def compute_distances(points):
    """Compute all pairwise distances using scipy."""
    return spd.pdist(points)

def smooth_min_max_ratio(points, beta):
    """
    Compute a smooth surrogate for (d_min / d_max)^2.
    Uses log-sum-exp for soft max and min.
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    max_d = np.max(dists)
    min_d = np.min(dists)
    
    exp_d = np.exp((dists - max_d) / beta)
    soft_max = max_d + beta * np.log(np.sum(exp_d))
    
    exp_neg_d = np.exp((min_d - dists) / beta)
    soft_min = min_d - beta * np.log(np.sum(exp_neg_d))
    
    if soft_max == 0:
        return 0.0
        
    ratio = soft_min / soft_max
    return ratio * ratio

def smooth_min_max_ratio_grad(points, beta):
    """
    Compute the smooth surrogate and its gradient.
    Returns (value, grad) where grad has shape (n, 2).
    """
    n = points.shape[0]
    # Compute pairwise distances
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0, np.zeros((n, 2))
    
    max_d = np.max(dists)
    min_d = np.min(dists)
    
    exp_d = np.exp((dists - max_d) / beta)
    soft_max = max_d + beta * np.log(np.sum(exp_d))
    
    exp_neg_d = np.exp((min_d - dists) / beta)
    soft_min = min_d - beta * np.log(np.sum(exp_neg_d))
    
    if soft_max == 0:
        return 0.0, np.zeros((n, 2))
        
    ratio = soft_min / soft_max
    value = ratio * ratio
    
    # Gradient computation
    # d(soft_max)/d(points[i]) and d(soft_min)/d(points[i])
    # For pdist, the gradient of distance between point i and j is (pi - pj) / d_ij
    
    grad = np.zeros((n, 2))
    
    # We need to compute gradients of soft_max and soft_min w.r.t. each point
    # soft_max = max_d + beta * log(sum(exp((dists - max_d)/beta)))
    # d(soft_max)/d(d_k) = exp((d_k - max_d)/beta) / sum(exp((dists - max_d)/beta))
    # d(soft_min)/d(d_k) = -exp((min_d - d_k)/beta) / sum(exp((min_d - dists)/beta))
    
    sum_exp_max = np.sum(exp_d)
    sum_exp_min = np.sum(exp_neg_d)
    
    # For each pair (i,j), the distance d_ij has gradient (pi - pj)/d_ij w.r.t. pi
    # and (pj - pi)/d_ij w.r.t. pj
    
    # Reconstruct full distance matrix
    dist_matrix = np.zeros((n, n))
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist_matrix[i, j] = dists[idx]
            dist_matrix[j, i] = dists[idx]
            idx += 1
    
    # Gradient of soft_max w.r.t. point p_i:
    # sum over j != i of w_max(j) * (p_i - p_j) / d_ij
    # where w_max(j) = exp((d_ij - max_d)/beta) / sum_exp_max
    
    # Gradient of soft_min w.r.t. point p_i:
    # sum over j != i of w_min(j) * (p_j - p_i) / d_ij
    # where w_min(j) = -exp((min_d - d_ij)/beta) / sum_exp_min
    
    # d(value)/d(p_i) = 2 * ratio * (d(soft_min)/d(p_i) * soft_max - soft_min * d(soft_max)/d(p_i)) / soft_max^2
    
    for i in range(n):
        grad_max_i = np.zeros(2)
        grad_min_i = np.zeros(2)
        
        for j in range(n):
            if i == j:
                continue
            d_ij = dist_matrix[i, j]
            if d_ij < 1e-12:
                continue
            
            # Weight for soft_max
            w_max = exp_d[np.where(np.triu_indices(n, k=1))[0] * (2*n - 1) // 2 + np.where(np.triu_indices(n, k=1))[1] - np.where(np.triu_indices(n, k=1))[1] * (np.where(np.triu_indices(n, k=1))[1] + 1) // 2] if False else 0
            
            # Simpler: just use the distance matrix directly
            # For pair (i,j), find its index in pdist
            if i < j:
                k = i * (2*n - i - 1) // 2 + (j - i - 1)
            else:
                k = j * (2*n - j - 1) // 2 + (i - j - 1)
            
            w_max_val = exp_d[k] / sum_exp_max
            w_min_val = exp_neg_d[k] / sum_exp_min
            
            diff = points[i] - points[j]
            unit_diff = diff / d_ij
            
            grad_max_i += w_max_val * unit_diff
            grad_min_i += w_min_val * (-unit_diff)  # d(soft_min)/d(d_k) is negative
        
        # d(value)/d(p_i) = 2 * ratio / soft_max^2 * (grad_min_i * soft_max - soft_min * grad_max_i)
        coeff = 2 * ratio / (soft_max * soft_max)
        grad[i] = coeff * (grad_min_i * soft_max - soft_min * grad_max_i)
    
    return value, grad

def complex_to_points(z):
    """Convert complex numbers to 2D points, with z[0] fixed at origin."""
    # z[0] is fixed at 0 (origin)
    # z[1:] are the other 15 points
    n = len(z)
    points = np.zeros((n, 2))
    for i in range(1, n):
        points[i, 0] = z[i].real
        points[i, 1] = z[i].imag
    return points

def points_to_complex(points):
    """Convert 2D points to complex numbers, with first point at origin."""
    n = len(points)
    z = np.zeros(n, dtype=complex)
    # Translate so first point is at origin
    offset = points[0]
    for i in range(1, n):
        z[i] = complex(points[i, 0] - offset[0], points[i, 1] - offset[1])
    return z

def objective_flat(x_flat, beta=0.1):
    """
    Objective function for flat representation of all 16 points.
    x_flat is a real array of length 32.
    """
    n = 16
    points = x_flat.reshape(n, 2)
    return -smooth_min_max_ratio(points, beta)

def objective_flat_grad(x_flat, beta=0.1):
    """
    Objective function and gradient for flat representation.
    """
    n = 16
    points = x_flat.reshape(n, 2)
    val, grad = smooth_min_max_ratio_grad(points, beta)
    return -val, -grad.flatten()

def lbfgsb_optimize(points, time_limit, beta=0.1, seed=None):
    """
    Optimize using L-BFGS-B with analytic gradient.
    """
    n = 16
    x_flat = points.flatten().copy()
    
    start_time = time.time()
    best_x_flat = x_flat.copy()
    best_score = calculate_score(points)
    
    # Use a callback to check time limit
    time_exceeded = [False]
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            time_exceeded[0] = True
            raise ValueError("Time limit exceeded")
        return False
    
    try:
        res = minimize(objective_flat, x_flat, 
                      method='L-BFGS-B',
                      jac=lambda x: objective_flat_grad(x, beta)[1],
                      callback=callback,
                      options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
        
        if res.success or not time_exceeded[0]:
            trial_score = calculate_score(res.x.reshape(n, 2))
            if trial_score > best_score:
                best_score = trial_score
                best_x_flat = res.x.copy()
    except (ValueError, Exception):
        pass
    
    best_points = best_x_flat.reshape(n, 2)
    return best_points, best_score

def local_search(points, time_limit, seed=None, use_smooth=False, beta=0.1):
    """
    Perform local search to maximize the score.
    Uses simulated annealing with adaptive step size.
    """
    n = points.shape[0]
    current_points = points.copy()
    
    if use_smooth:
        current_score = smooth_min_max_ratio(current_points, beta)
    else:
        current_score = calculate_score(current_points)
        
    best_points = current_points.copy()
    best_score = current_score
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    start_time = time.time()
    temp = 0.05
    min_temp = 1e-6
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        idx = rng.randint(n)
        
        # Adaptive step size based on current temperature
        step_size = 0.1 * temp + 0.005
        perturbation = rng.randn(2) * step_size
        
        new_point = current_points[idx].copy()
        new_point += perturbation
        
        trial_points = current_points.copy()
        trial_points[idx] = new_point
        
        if use_smooth:
            trial_score = smooth_min_max_ratio(trial_points, beta)
        else:
            trial_score = calculate_score(trial_points)
            
        if trial_score >= current_score:
            current_points = trial_points
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = current_score - trial_score
            if delta < 1e-10:
                current_points = trial_points
                current_score = trial_score
            else:
                if rng.rand() < np.exp(-delta / (temp + 1e-10)):
                    current_points = trial_points
                    current_score = trial_score
        
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp
            
        if time.time() - start_time > time_limit * 0.95:
            break

    final_score = calculate_score(best_points)
    return best_points, final_score

def nelder_mead_polish(points, time_limit, seed=None):
    """
    Use Nelder-Mead on the smooth surrogate to polish the points.
    """
    n = points.shape[0]
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        return -smooth_min_max_ratio(pts, beta=0.03)
        
    start_time = time.time()
    best_pts = points.copy()
    best_score = calculate_score(best_pts)
    
    # Time-based callback
    time_exceeded = [False]
    def callback(xk):
        if time.time() - start_time > time_limit:
            time_exceeded[0] = True
            raise ValueError("Time limit exceeded")
        return False
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       callback=callback,
                       options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12, 'initial_simplex': 1e-3})
        
        if res.success or not time_exceeded[0]:
            trial_pts = res.x.reshape(n,2)
            trial_score = calculate_score(trial_pts)
            if trial_score > best_score:
                best_pts = trial_pts
                best_score = trial_score
    except (ValueError, Exception):
        pass
        
    return best_pts, best_score

def optimize_geometry(points, time_limit, seed=None):
    """
    A structured optimizer combining local search and Nelder-Mead polish.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    start_time = time.time()
    
    while time.time() - start_time < time_limit - 3:
        remaining = remaining_time_calc(start_time, time_limit) - 3
        if remaining < 5:
            break
        
        # Phase 1: L-BFGS-B with analytic gradient (most effective)
        chunk1 = min(remaining * 0.35, 25)
        if chunk1 > 3:
            res_points, res_score = lbfgsb_optimize(current_points, chunk1, beta=0.08, seed=seed)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
        
        # Phase 2: Smooth local search
        remaining = remaining_time_calc(start_time, time_limit) - 3
        if remaining < 5:
            break
        chunk2 = min(remaining * 0.25, 15)
        if chunk2 > 3:
            res_points, res_score = local_search(current_points, chunk2, seed=seed+100, use_smooth=True, beta=0.08)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Phase 3: Nelder-Mead polish
        remaining = remaining_time_calc(start_time, time_limit) - 3
        if remaining < 5:
            break
        chunk3 = min(remaining * 0.25, 15)
        if chunk3 > 3:
            res_points, res_score = nelder_mead_polish(current_points, chunk3, seed=seed+200)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Phase 4: Exact local search
        remaining = remaining_time_calc(start_time, time_limit) - 3
        if remaining < 5:
            break
        chunk4 = min(remaining * 0.15, 10)
        if chunk4 > 2:
            res_points, res_score = local_search(current_points, chunk4, seed=seed+300, use_smooth=False)
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                current_points = best_points.copy()
                
        # Perturb for next restart (gentler perturbations)
        num_perturb = rng.randint(1, 3)
        indices = rng.choice(n, num_perturb, replace=False)
        for idx in indices:
            current_points[idx] += rng.randn(2) * 0.08
            
    return best_points, best_score

def remaining_time_calc(start_time, total_limit):
    return total_limit - (time.time() - start_time)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    global_deadline = time.time() + 625
    
    candidates = []
    
    # Candidate 1: 4x4 Square Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    candidates.append(grid_points)
    
    # Candidate 2: Points on two concentric circles
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(np.pi/8, 2*np.pi, 8, endpoint=False)
    r1 = 1.0
    r2 = 0.5
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    circle_points = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    candidates.append(circle_points)
    
    # Candidate 3: Random points in a disk (multiple seeds)
    for i in range(3):
        rng_init = np.random.RandomState(42 + i * 100)
        angles = rng_init.rand(n) * 2 * np.pi
        radii = np.sqrt(rng_init.rand(n)) * 1.5
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        candidates.append(np.column_stack((x, y)))
    
    # Candidate 4: Hexagonal lattice points (first 16 in a hex grid)
    hex_grid = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = np.sqrt(3)/2 * r
            hex_grid.append([x, y])
    hex_grid = np.array(hex_grid)
    dists_to_origin = np.linalg.norm(hex_grid, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    hex_candidate = hex_grid[idx]
    candidates.append(hex_candidate)
    
    # Candidate 5: Symmetric configuration with 1 center + 6 + 9
    center = np.array([0.0, 0.0])
    ring1_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
    r2 = np.sqrt(3)
    ring2 = r2 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    symmetric_points = np.vstack([center, ring1, ring2])
    candidates.append(symmetric_points)
    
    # Candidate 6: 16 points on a circle (uniform)
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    circle_uniform = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(circle_uniform)
    
    best_points = None
    best_score = 0
    
    num_candidates = len(candidates)
    
    # Allocate time: give more time to later candidates if they're promising
    # Use adaptive time allocation
    base_time = 30
    
    for i, cand in enumerate(candidates):
        if time.time() > global_deadline - 30:
            break
            
        cand = cand.astype(float)
        
        # Give less time to early candidates, more to later ones
        if i < num_candidates // 2:
            time_per_candidate = base_time
        else:
            time_per_candidate = base_time + 10
        
        # Ensure we don't exceed deadline
        max_allowed = global_deadline - 30 - time.time()
        time_per_candidate = min(time_per_candidate, max_allowed)
        
        if time_per_candidate < 5:
            continue
        
        res_points, res_score = optimize_geometry(cand, time_per_candidate, seed=1000+i)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # Final optimization with remaining time
    if best_points is not None and time.time() < global_deadline - 15:
        remaining_time = global_deadline - 15 - time.time()
        if remaining_time > 15:
            res_points, res_score = optimize_geometry(best_points, remaining_time, seed=9999)
            if res_score > best_score:
                best_points = res_points
                best_score = res_score

    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
        
    return best_points

# EVOLVE-BLOCK-END
