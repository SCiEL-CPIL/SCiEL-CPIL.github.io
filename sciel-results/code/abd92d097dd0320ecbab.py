"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(pos):
    """
    Computes the maximum radius for the given positions.
    Returns 0 if invalid.
    """
    if pos.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    # Check if inside [-0.5, 0.5]
    if np.any(pos < -0.5) or np.any(pos > 0.5):
        return 0.0

    # Pairwise distances
    d = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = dist.min()
    
    # Distance to walls
    min_wall_dist = (0.5 - np.abs(pos)).min()
    
    return min(0.5 * min_pair_dist, min_wall_dist)

def _violation(pos, r):
    """
    Compute violation for a given radius r.
    Returns a positive value if constraints are violated, 0 if feasible.
    Violation = max(0, 2r - min_pair_dist, r - min_wall_dist)
    """
    if pos.shape != (N, 2):
        return np.inf
    if not np.all(np.isfinite(pos)):
        return np.inf
    
    # Pairwise distances
    d = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = dist.min()
    
    # Distance to walls
    min_wall_dist = (0.5 - np.abs(pos)).min()
    
    # Violation terms
    v1 = 2.0 * r - min_pair_dist
    v2 = r - min_wall_dist
    
    return max(0.0, v1, v2)

def _violation_obj(flat_pos, r, beta=50.0):
    """
    Objective function for L-BFGS-B: minimize violation with soft-min penalty.
    We want to minimize the maximum violation.
    Use soft-min to make it differentiable.
    """
    pos = flat_pos.reshape(N, 2)
    
    # Pairwise distances
    d = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall_x = 0.5 - np.abs(pos[:, 0])
    wall_y = 0.5 - np.abs(pos[:, 1])
    
    # Violation terms for each constraint:
    # For pair (i,j): 2r - dist[i,j]
    # For wall x: r - (0.5 - |x_i|)
    # For wall y: r - (0.5 - |y_i|)
    
    # We want to minimize max of these violations.
    # Use soft-max: max(v) ≈ 1/beta * log(sum(exp(beta * v)))
    
    # Collect all violation terms
    triu_i, triu_j = np.triu_indices(N, k=1)
    pair_violations = 2.0 * r - dist[triu_i, triu_j]
    
    wall_x_violations = r - wall_x
    wall_y_violations = r - wall_y
    
    all_violations = np.concatenate([pair_violations, wall_x_violations, wall_y_violations])
    
    # Soft-max with log-sum-exp trick
    m = np.max(all_violations)
    if m == -np.inf:
        return 0.0
    shifted = beta * (all_violations - m)
    # Clip to prevent overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum_exp = m + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    # We want to minimize the soft-max of violations
    # But we also need to handle the case where all violations are negative (feasible)
    # In that case, soft-max will be negative, which is good.
    # L-BFGS-B will try to minimize this.
    
    return log_sum_exp

def _violation_grad(flat_pos, r, beta=50.0):
    """
    Gradient of the soft-max violation objective.
    """
    pos = flat_pos.reshape(N, 2)
    
    # Pairwise distances
    d = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall_x = 0.5 - np.abs(pos[:, 0])
    wall_y = 0.5 - np.abs(pos[:, 1])
    
    # Violation terms
    triu_i, triu_j = np.triu_indices(N, k=1)
    pair_violations = 2.0 * r - dist[triu_i, triu_j]
    
    wall_x_violations = r - wall_x
    wall_y_violations = r - wall_y
    
    all_violations = np.concatenate([pair_violations, wall_x_violations, wall_y_violations])
    
    # Soft-max probabilities
    m = np.max(all_violations)
    shifted = beta * (all_violations - m)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    p = exp_shifted / sum_exp
    
    # Gradient of soft-max w.r.t each violation term:
    # d/dv_k (1/beta * log(sum exp(beta * v_j))) = exp(beta * v_k) / sum(exp(beta * v_j)) = p_k
    # So dObj/dv_k = p_k
    
    # Now we need d(v_k)/d(pos)
    # For pair (i,j): v = 2r - dist[i,j]
    # d(v)/d(pos[i]) = -d(dist[i,j])/d(pos[i]) = -(pos[i] - pos[j]) / dist[i,j]
    # d(v)/d(pos[j]) = -d(dist[i,j])/d(pos[j]) = -(pos[j] - pos[i]) / dist[i,j]
    
    # For wall x: v = r - (0.5 - |x_i|) = r - 0.5 + |x_i|
    # d(v)/d(x_i) = sign(x_i)
    
    # For wall y: v = r - (0.5 - |y_i|) = r - 0.5 + |y_i|
    # d(v)/d(y_i) = sign(y_i)
    
    grad = np.zeros((N, 2))
    
    # Pairwise gradients
    diff = pos[triu_i] - pos[triu_j]
    safe_dists = np.where(dist[triu_i, triu_j] < 1e-12, 1e-12, dist[triu_i, triu_j])
    unit_vecs = diff / safe_dists[:, None]
    
    p_pairs = p[:len(triu_i)]
    
    # d(v_pair)/d(pos[i]) = -unit_vec
    # d(v_pair)/d(pos[j]) = +unit_vec
    np.add.at(grad, triu_i, -p_pairs[:, None] * unit_vecs)
    np.add.at(grad, triu_j, p_pairs[:, None] * unit_vecs)
    
    # Wall gradients
    p_wall_x = p[len(triu_i):len(triu_i)+N]
    p_wall_y = p[len(triu_i)+N:]
    
    sign_x = np.sign(pos[:, 0])
    sign_y = np.sign(pos[:, 1])
    sign_x = np.where(sign_x == 0, 0, sign_x)
    sign_y = np.where(sign_y == 0, 0, sign_y)
    
    grad[:, 0] += p_wall_x * sign_x
    grad[:, 1] += p_wall_y * sign_y
    
    return grad

def _grid_init():
    """Generate a square grid initial layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_init():
    """Generate a hexagonal-like initial layout."""
    pos = []
    # Hexagonal packing
    dx = 1.0 / 7.5
    dy = dx * np.sqrt(3) / 2
    
    for row in range(8):
        for col in range(8):
            x = (col + 0.5 * (row % 2)) * dx - 0.5
            y = row * dy - 0.5 + dy / 2
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                pos.append((x, y))
    
    pos = np.array(pos[:N])
    if len(pos) < N:
        # Fill remaining with random
        rng = np.random.default_rng(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(pos), 2))
        pos = np.vstack([pos, extra])
    
    return pos

def _random_init(seed):
    """Generate a random initial layout."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4, 0.4, size=(N, 2))

def _lbfgs_polish(pos, r, time_budget, beta=50.0):
    """
    Polish a configuration using L-BFGS-B to minimize violation for radius r.
    Returns the polished position.
    """
    t_start = time.time()
    x0 = pos.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        return _violation_obj(x, r, beta)
        
    def grad(x):
        return _violation_grad(x, r, beta)
    
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200})
        if res.success or res.x is not None:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return pos

def _binary_search_radius(pos_init, r_low, r_high, time_budget):
    """
    Binary search for the maximum feasible radius.
    For each candidate radius, use L-BFGS-B to minimize violation.
    If violation is below tolerance, the radius is feasible.
    """
    t_start = time.time()
    best_pos = pos_init.copy()
    best_r = r_low
    
    # Binary search
    for _ in range(15):  # Max 15 iterations
        if time.time() - t_start > time_budget:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        pos_candidate = pos_init.copy()
        # Run L-BFGS-B to minimize violation for r_mid
        pos_candidate = _lbfgs_polish(pos_candidate, r_mid, min(5.0, time_budget - (time.time() - t_start)))
        
        # Check if feasible
        viol = _violation(pos_candidate, r_mid)
        
        if viol < 1e-8:
            # Feasible! Update best and increase lower bound
            r_low = r_mid
            best_r = r_mid
            best_pos = pos_candidate.copy()
        else:
            # Not feasible, decrease upper bound
            r_high = r_mid
        
        # If the interval is small enough, stop
        if r_high - r_low < 1e-6:
            break
    
    return best_pos, best_r

def _simulated_annealing(pos, time_budget, r_target=None):
    """
    Simulated annealing to improve the packing.
    If r_target is given, try to achieve that radius.
    Otherwise, maximize the radius.
    """
    t_start = time.time()
    cur_pos = pos.copy()
    cur_r = _compute_radius(cur_pos)
    best_pos = cur_pos.copy()
    best_r = cur_r
    
    if r_target is None:
        r_target = cur_r * 1.05  # Try to improve by 5%
    
    # Annealing parameters
    T_start = 0.01
    T_end = 1e-6
    alpha = 0.999
    
    T = T_start
    
    while time.time() - t_start < time_budget and T > T_end:
        # Pick a random circle to move
        i = np.random.randint(N)
        old = cur_pos[i].copy()
        
        # Random perturbation
        step = 0.005 * (T / T_start) ** 0.5
        cur_pos[i] = old + np.random.normal(0, step, size=2)
        cur_pos[i] = np.clip(cur_pos[i], -0.5, 0.5)
        
        new_r = _compute_radius(cur_pos)
        
        # Acceptance criterion
        if new_r >= cur_r:
            cur_r = new_r
            if cur_r > best_r:
                best_r = cur_r
                best_pos = cur_pos.copy()
        else:
            # Metropolis acceptance
            delta_r = cur_r - new_r
            if np.random.rand() < np.exp(-delta_r / (T + 1e-10)):
                cur_r = new_r
            else:
                cur_pos[i] = old
        
        T *= alpha
        
        # Periodic cooling
        if np.random.rand() < 0.01:
            T *= 0.9
    
    return best_pos, best_r

def construct():
    t_start = time.time()
    deadline = 140.0  # Safety margin (150s limit)
    
    # 1. Generate multiple initial layouts
    inits = {
        'grid': _grid_init(),
        'hex': _hex_init(),
        'rand1': _random_init(42),
        'rand2': _random_init(123),
    }
    
    best_pos = None
    best_r = 0.0
    
    # 2. For each initial layout, try to improve
    for name, pos_init in inits.items():
        if time.time() - t_start > deadline - 20:
            break
        
        r_init = _compute_radius(pos_init)
        if r_init > best_r:
            best_r = r_init
            best_pos = pos_init.copy()
        
        # 3. Binary search for better radius
        # Start with a conservative low bound
        r_low = r_init
        r_high = r_init * 1.15  # Try up to 15% improvement
        
        remaining = deadline - 10 - (time.time() - t_start)
        if remaining > 10:
            pos_bs, r_bs = _binary_search_radius(pos_init, r_low, r_high, min(20.0, remaining))
            if r_bs > best_r:
                best_r = r_bs
                best_pos = pos_bs.copy()
    
    # 4. Final polish with simulated annealing
    remaining = deadline - 5 - (time.time() - t_start)
    if remaining > 5 and best_pos is not None:
        pos_sa, r_sa = _simulated_annealing(best_pos, min(15.0, remaining), r_target=None)
        if r_sa > best_r:
            best_r = r_sa
            best_pos = pos_sa.copy()
    
    # Final validation
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        return _grid_init()
    
    # Ensure within bounds
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    return best_pos

# EVOLVE-BLOCK-END
