# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """
    Calculate the metric R = (d_min / d_max)^2
    """
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

def smooth_objective(flat_points, beta=100.0):
    """
    Smooth surrogate: minimize -log(sum(exp(-beta*d_i))) + log(max(d_i))
    This approximates minimizing -log(d_min) + log(d_max) = log(d_max/d_min)
    """
    points = flat_points.reshape(-1, 2)
    n = points.shape[0]
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Only upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    if len(dists) == 0:
        return 1e10
    
    # Softmin approximation: -log(sum(exp(-beta*d))) / beta ≈ min(d)
    # We want to maximize d_min, so minimize -d_min ≈ log(sum(exp(-beta*d)))/beta
    # And minimize d_max, so minimize log(max(d))
    
    # For numerical stability
    max_d = np.max(dists)
    if max_d < 1e-10:
        return 1e10
    
    # Softmin term: approximates -d_min
    exp_term = np.exp(-beta * dists)
    log_sum_exp = np.log(np.sum(exp_term))
    softmin_neg = log_sum_exp / beta  # This is approximately -log(sum(exp(-beta*d)))/(-beta) ... let me rethink
    
    # Actually: min(d) ≈ (1/beta) * log(sum(exp(-beta*d)))... no
    # min(d) ≈ -(1/beta) * log(sum(exp(-beta*d)))
    # So -min(d) ≈ (1/beta) * log(sum(exp(-beta*d)))
    
    softmin_term = (1.0 / beta) * np.log(np.sum(np.exp(-beta * dists)))
    
    # For max: use log-sum-exp
    log_max = max_d + np.log(np.sum(np.exp(dists - max_d)))
    
    # We want to minimize log(d_max) - log(d_min) = log(d_max/d_min)
    # ≈ log_max - (-softmin_term) = log_max + softmin_term
    # Wait: -log(d_min) ≈ softmin_term (since d_min ≈ -(1/beta)*log(sum(exp(-beta*d))))
    # So log(d_max) - log(d_min) ≈ log_max + softmin_term
    
    objective = log_max + softmin_term
    return objective

def smooth_objective_grad(flat_points, beta=100.0):
    """
    Gradient of the smooth objective.
    """
    points = flat_points.reshape(-1, 2)
    n = points.shape[0]
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(np.maximum(dists_sq[iu], 1e-20))
    
    if len(dists) == 0:
        return 1e10, np.zeros_like(flat_points)
    
    max_d = np.max(dists)
    if max_d < 1e-10:
        return 1e10, np.zeros_like(flat_points)
    
    # Softmin term gradient
    exp_term = np.exp(-beta * dists)
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp  # softmax weights for min
    
    # Gradient of (1/beta)*log(sum(exp(-beta*d_i))) w.r.t. points
    # = (1/beta) * (1/sum_exp) * sum_j (-beta * exp(-beta*d_j) * dd_j/dp)
    # = -sum_j weights_j * dd_j/dp
    
    # dd_i/dp_k where d_i = ||p_a - p_b||
    # dd_i/dp_k = (p_k - p_m)/d_i if k==a or k==b, else 0
    # where (a,b) is the pair for distance i
    
    grad = np.zeros_like(points)
    
    # For each distance i in upper triangle:
    # pair index: iu[0][i], iu[1][i]
    # weight: weights[i]
    # contribution to grad[iu[0][i]]: -weights[i] * (p[iu[0][i]] - p[iu[1][i]]) / d_i
    # contribution to grad[iu[1][i]]: -weights[i] * (p[iu[1][i]] - p[iu[0][i]]) / d_i
    
    for idx in range(len(dists)):
        a = iu[0][idx]
        b = iu[1][idx]
        d = dists[idx]
        w = weights[idx]
        if d < 1e-10:
            continue
        direction = (points[a] - points[b]) / d
        grad[a] -= w * direction
        grad[b] += w * direction
    
    # Log max term gradient
    # log(max(d)) - use log-sum-exp: log(sum(exp(d_i - max_d))) + max_d
    # Gradient of log-sum-exp: (1/sum(exp(d-max_d))) * sum(exp(d_i - max_d) * dd_i/dp)
    shifted = dists - max_d
    exp_shifted = np.exp(shifted)
    sum_exp_shifted = np.sum(exp_shifted)
    lse_weights = exp_shifted / sum_exp_shifted
    
    for idx in range(len(dists)):
        a = iu[0][idx]
        b = iu[1][idx]
        d = dists[idx]
        w = lse_weights[idx]
        if d < 1e-10:
            continue
        direction = (points[a] - points[b]) / d
        grad[a] += w * direction
        grad[b] -= w * direction
    
    objective = max_d + np.log(sum_exp_shifted) + (1.0 / beta) * np.log(sum_exp)
    return objective, grad.flatten()

def local_search_adaptive(points, time_limit, initial_step=0.1):
    """
    Adaptive step local search to escape local minima.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = calculate_score(best_points)
    current_points = points.copy()
    current_score = best_score
    step_size = initial_step
    n_iter = 0
    
    while time.time() < start_time + time_limit:
        n_iter += 1
        # Generate a random perturbation
        noise = np.random.randn(*current_points.shape) * step_size
        candidate = current_points + noise
        
        candidate_score = calculate_score(candidate)
        
        if candidate_score > current_score:
            current_points = candidate
            current_score = candidate_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate.copy()
            step_size *= 1.05  # Slightly increase step
        else:
            step_size *= 0.95  # Decrease step
            step_size = max(step_size, 1e-6)
        
        # Periodic reset to best if stuck
        if n_iter % 100 == 0 and time.time() < start_time + time_limit - 5:
            if current_score < best_score * 0.99:
                current_points = best_points + np.random.randn(*best_points.shape) * step_size * 0.5
                current_score = calculate_score(current_points)
    
    return best_points

def generate_hex_lattice(n=16):
    """Generate points on a hexagonal lattice."""
    pts = []
    # 4x4 hex lattice
    for i in range(4):
        for j in range(4):
            x = j + (0.5 if i % 2 else 0)
            y = i * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    # Normalize
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    return pts

def generate_circular_ring(n=16, radius=1.0):
    """Generate points on a circle."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return pts

def refine_points(points, time_limit):
    """
    Refine points using multiple optimization strategies.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = calculate_score(best_points)
    
    # Strategy 1: L-BFGS-B with smooth objective
    if time.time() < start_time + time_limit - 30:
        try:
            flat_points = points.flatten()
            result = minimize(
                lambda x: smooth_objective(x, beta=50.0),
                flat_points,
                jac=lambda x: smooth_objective_grad(x, beta=50.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < 1e10:
                candidate = result.x.reshape(-1, 2)
                score = calculate_score(candidate)
                if score > best_score:
                    best_points = candidate
                    best_score = score
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead on d_max/d_min
    if time.time() < start_time + time_limit - 20:
        try:
            def obj_nm(flat):
                pts = flat.reshape(-1, 2)
                dists = spd.pdist(pts)
                if len(dists) == 0:
                    return 1e10
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_min < 1e-10:
                    return 1e10
                return d_max / d_min
            
            flat_points = best_points.flatten()
            result = minimize(
                obj_nm,
                flat_points,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if result.fun < 1e10:
                candidate = result.x.reshape(-1, 2)
                score = calculate_score(candidate)
                if score > best_score:
                    best_points = candidate
                    best_score = score
        except Exception:
            pass
    
    # Strategy 3: Local search with adaptive steps
    remaining = max(0, start_time + time_limit - 10 - time.time())
    if remaining > 5:
        try:
            candidate = local_search_adaptive(best_points, remaining, initial_step=0.05)
            score = calculate_score(candidate)
            if score > best_score:
                best_points = candidate
                best_score = score
        except Exception:
            pass
    
    return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    # Deadline guard: 630s limit, use 600s for safety
    deadline = time.time() + 590
    
    # Generate initializations
    inits = []
    
    # 1. Hexagonal lattice
    inits.append(generate_hex_lattice(n))
    
    # 2. Circular ring
    inits.append(generate_circular_ring(n))
    
    # 3. Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    points_grid = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(points_grid)
    
    # 4. Random initializations with different seeds
    for seed in [42, 123, 777, 2024]:
        np.random.seed(seed)
        pts = np.random.randn(n, d)
        pts -= np.mean(pts, axis=0)
        inits.append(pts)
    
    # 5. Two concentric rings
    angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = np.column_stack([0.5*np.cos(angles_inner), 0.5*np.sin(angles_inner)])
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inits.append(np.vstack([inner_pts, outer_pts]))
    
    best_points = inits[0]
    best_score = calculate_score(best_points)
    
    # Refine each initialization
    for init_pts in inits:
        if time.time() > deadline - 15:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - 15 - time.time()
        per_init_time = min(60, remaining / max(1, len(inits)))
        
        res_points = refine_points(init_pts, per_init_time)
        res_score = calculate_score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # Final polish on the best result
    if time.time() < deadline - 10:
        remaining_time = deadline - 10 - time.time()
        best_points = refine_points(best_points, remaining_time)
        
    return best_points

# EVOLVE-BLOCK-END
