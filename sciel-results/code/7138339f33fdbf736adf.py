import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _build_grid_init(nx, ny, nz, noise_scale=0.0, seed=42):
    """Build a grid initialization."""
    if nx == 1: xs = np.array([0.0])
    else: xs = np.linspace(-0.5, 0.5, nx)
    
    if ny == 1: ys = np.array([0.0])
    else: ys = np.linspace(-0.5, 0.5, ny)
    
    if nz == 1: zs = np.array([0.0])
    else: zs = np.linspace(-0.5, 0.5, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    if len(centers) != N:
        if len(centers) < N:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
        else:
            centers = centers[:N]

    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        centers += rng.normal(0, noise_scale, centers.shape)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(seed=42):
    """
    Generate N points using a Halton sequence (low discrepancy).
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    bases = [2, 3, 5]
    points = []
    for i in range(1, N + 1):
        pt = [halton(i, b) for b in bases]
        pt_mapped = [p - 0.5 for p in pt]
        points.append(pt_mapped)
        
    centers = np.array(points)
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _soft_min_objective(x, t, beta):
    """
    Soft-min objective using log-sum-exp for smoothness.
    x: flattened centers (N*3,)
    t: radius variable
    beta: temperature parameter for soft-min
    
    We want to maximize t subject to:
    1. 0.5 - |c_ik| >= t  for all i, k
    2. |c_i - c_j| >= 2t  for all i < j
    
    Using soft-min: -soft_min(values) where soft_min approximates min(values)
    Actually, we want to maximize the minimum clearance.
    Let's use: maximize t - (1/beta) * log(sum(exp(-beta * margin)))
    where margin is the minimum of all clearances.
    
    Alternatively, minimize: -t + (1/beta) * log(sum(exp(-beta * margin_i)))
    where margin_i are the individual clearances.
    """
    centers = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    # Wall margins: 0.5 - |c_ik|
    wall_margins = 0.5 - np.abs(centers)
    
    # Pair margins: |c_i - c_j| / 2
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_margins = dists / 2.0
    # Only consider i < j to avoid double counting
    pair_margins = pair_margins[np.triu_indices(N, k=1)]
    
    # All margins
    margins = np.concatenate([wall_margins.ravel(), pair_margins])
    
    # Soft-min approximation: -1/beta * log(sum(exp(-beta * margin)))
    # This approximates min(margins) as beta -> infinity
    # We want to maximize min(margins), so minimize -soft_min
    # soft_min = -1/beta * log(sum(exp(-beta * margin)))
    # So we minimize: -t + 1/beta * log(sum(exp(-beta * margin)))
    #                 = -t + 1/beta * log(sum(exp(-beta * margin)))
    
    # For numerical stability
    max_margin = np.max(margins)
    exp_terms = np.exp(-beta * (margins - max_margin))
    log_sum_exp = np.log(np.sum(exp_terms)) + (-beta * max_margin)
    soft_min_approx = -log_sum_exp / beta
    
    # Objective: minimize -t + penalty for t > soft_min
    # Actually, we want t <= min(margins), so we penalize t > soft_min
    # Use: -t + max(0, t - soft_min_approx)^2 * weight
    # Or simpler: -t + (1/beta) * log(sum(exp(-beta * (margin - t))))
    # This is the soft version of: -t + max(0, t - min(margin))
    
    # Let's use: minimize -t + (1/beta) * log(sum(exp(-beta * (margin - t))))
    # This approximates -t + max(0, t - min(margin))
    
    margin_minus_t = margins - t
    max_mmt = np.max(margin_minus_t)
    exp_mmt = np.exp(-beta * (margin_minus_t - max_mmt))
    log_sum_exp_mmt = np.log(np.sum(exp_mmt)) + (-beta * max_mmt)
    soft_penalty = log_sum_exp_mmt / beta
    
    return -t + soft_penalty

def _soft_min_gradient(x, t, beta):
    """
    Gradient of the soft-min objective.
    """
    centers = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    grad = np.zeros_like(x)
    
    # Wall margins: 0.5 - |c_ik|
    wall_margins = 0.5 - np.abs(centers)
    
    # Pair margins: |c_i - c_j| / 2
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_margins = dists / 2.0
    pair_margins = pair_margins[np.triu_indices(N, k=1)]
    
    margins = np.concatenate([wall_margins.ravel(), pair_margins])
    
    # margin_minus_t = margins - t
    margin_minus_t = margins - t
    max_mmt = np.max(margin_minus_t)
    exp_mmt = np.exp(-beta * (margin_minus_t - max_mmt))
    sum_exp = np.sum(exp_mmt)
    weights = exp_mmt / sum_exp  # Soft-max weights
    
    # Gradient of -t: -1 for t
    grad[N*3] = -1.0
    
    # Gradient of soft_penalty w.r.t. t: -sum(weights) = -1
    grad[N*3] += -1.0  # Wait, let's recompute
    
    # soft_penalty = (1/beta) * log(sum(exp(-beta * (margin - t))))
    # d(soft_penalty)/dt = (1/beta) * (1/sum(exp)) * sum(exp * (-beta) * (-1))
    #                    = (1/beta) * (1/sum(exp)) * sum(exp) * beta
    #                    = 1
    # So gradient w.r.t. t is -1 (from -t) + 1 (from soft_penalty) = 0?
    # That doesn't seem right. Let me re-derive.
    
    # f = -t + (1/beta) * log(sum_j exp(-beta * (m_j - t)))
    # df/dt = -1 + (1/beta) * (1/sum_j exp(-beta*(m_j-t))) * sum_j exp(-beta*(m_j-t)) * beta
    #       = -1 + 1 = 0
    # That's because the soft_penalty exactly compensates the -t term in the gradient.
    # The gradient w.r.t. t is actually 0, which means we need to look at the
    # constraint more carefully.
    
    # Actually, the issue is that the soft penalty is designed to be 0 when t <= min(margins)
    # and positive when t > min(margins). The gradient w.r.t. t should be 0 when feasible
    # and negative when infeasible (to push t down).
    
    # Let me use a different formulation:
    # Minimize: -t + (1/beta) * sum_j log(1 + exp(-beta * (m_j - t)))
    # This is a smooth approximation of -t + max(0, t - min(m_j))
    
    # For now, let's use a simpler penalty approach.
    
    # Gradient w.r.t. centers:
    # d(soft_penalty)/d(c_ik) = (1/beta) * (1/sum_exp) * sum_j exp(-beta*(m_j-t)) * (-beta) * dm_j/dc_ik
    #                         = -sum_j weights_j * dm_j/dc_ik
    
    # For wall margins: m_j = 0.5 - |c_ik|, dm_j/dc_ik = -sign(c_ik)
    # For pair margins: m_j = |c_i - c_j|/2, dm_j/dc_i = (c_i - c_j)/(2*|c_i-c_j|)
    
    # This is getting complex. Let's use a simpler penalty method instead.
    return grad

def _penalty_objective(x, target_r):
    """
    Penalty-based objective: minimize sum of squared violations.
    x: flattened array of shape (N*3,)
    target_r: the radius we want to achieve
    """
    centers = x.reshape(N, 3)
    
    # Wall constraints: 0.5 - |c_ik| >= target_r
    wall_violations = np.maximum(0.0, target_r - (0.5 - np.abs(centers)))
    
    # Pairwise constraints: |c_i - c_j| >= 2*target_r
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_violations = np.maximum(0.0, 2.0 * target_r - dists)
    
    # Sum of squared violations
    obj = np.sum(wall_violations**2) + np.sum(pair_violations**2)
    
    return obj

def _penalty_gradient(x, target_r):
    """
    Analytical gradient of the penalty objective.
    """
    centers = x.reshape(N, 3)
    grad = np.zeros_like(x)
    
    # Wall constraint gradient
    wall_margin = 0.5 - np.abs(centers)
    wall_violations = np.maximum(0.0, target_r - wall_margin)
    
    if np.any(wall_violations > 0):
        wall_grad = -2.0 * wall_violations * np.sign(centers)
        grad += wall_grad.flatten()
    
    # Pairwise constraint gradient
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pair_violations = np.maximum(0.0, 2.0 * target_r - dists)
    
    for i in range(N):
        for j in range(i+1, N):
            if pair_violations[i, j] > 0:
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    dir_vec = diff / dist
                    coeff = 2.0 * pair_violations[i, j]
                    grad[i*3:i*3+3] -= coeff * dir_vec
                    grad[j*3:j*3+3] += coeff * dir_vec
    
    return grad

def _optimize_penalty(centers, deadline):
    """
    Optimize using penalty method with L-BFGS-B.
    Returns (optimized_centers, achieved_radius).
    """
    x0 = centers.flatten()
    current_r = _compute_radius(centers)
    
    best_x = x0.copy()
    best_r = current_r
    
    target_r = current_r * 1.001
    
    while time.time() < deadline - 2.0 and target_r < 0.12:
        result = minimize(
            _penalty_objective,
            x0,
            args=(target_r,),
            method='L-BFGS-B',
            jac=_penalty_gradient,
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        new_centers = result.x.reshape(N, 3)
        new_r = _compute_radius(new_centers)
        
        if new_r >= target_r - 1e-8:
            best_x = result.x.copy()
            best_r = new_r
            target_r = min(target_r * 1.005, 0.12)
            x0 = result.x.copy()
        else:
            if new_r > best_r:
                best_r = new_r
                best_x = result.x.copy()
            break
    
    return best_x.reshape(N, 3), best_r

def _local_optimize(centers, deadline):
    """
    Local optimization: identify limiting constraint and move points.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    while time.time() < deadline - 1.0:
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        i_wall, j_wall = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        if min_dist / 2.0 < min_wall:
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-9:
                dir_i = diff / norm
                dir_j = -diff / norm
            else:
                dir_i = np.random.normal(0, 1, 3)
                dir_i /= np.linalg.norm(dir_i)
                dir_j = -dir_i
                
            step = 0.005
            centers[i] += step * dir_i
            centers[j] += step * dir_j
        else:
            k = j_wall
            dir = -np.sign(centers[i_wall, k])
            if dir == 0:
                dir = 1.0
            step = 0.005
            centers[i_wall, k] += step * dir
            
        centers = np.clip(centers, -0.5, 0.5)
        
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = _get_deadline()
    
    best_centers = None
    best_radius = 0.0
    
    initializations = []
    
    grid_configs = [
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 4),
        (10, 5, 2),
        (5, 10, 2),
        (5, 2, 10),
    ]
    
    for nx, ny, nz in grid_configs:
        for noise in [0.0, 0.01, 0.02]:
            centers = _build_grid_init(nx, ny, nz, noise, seed=nx*100+ny*10+nz)
            initializations.append(centers)
            
    initializations.append(_build_halton_init())
    
    rng = np.random.default_rng(123)
    for _ in range(5):
        centers = rng.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
        
    n_inits = len(initializations)
    time_per_init = (_get_deadline() - time.time() - 5.0) / n_inits
    
    for centers in initializations:
        if time.time() > deadline - 5.0:
            break
            
        opt_centers, opt_radius = _local_optimize(centers, time.time() + time_per_init * 0.3)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
            
        if time.time() < deadline - 5.0:
            penalty_deadline = min(time.time() + time_per_init * 0.7, deadline - 5.0)
            pen_centers, pen_radius = _optimize_penalty(opt_centers, penalty_deadline)
            
            if pen_radius > best_radius:
                best_radius = pen_radius
                best_centers = pen_centers

    if best_centers is None:
        best_centers = _build_grid_init(5, 5, 4, 0.0)
        
    return best_centers
