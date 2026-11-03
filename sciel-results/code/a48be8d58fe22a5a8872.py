"""Circle packing in unit square - 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _dist_matrix(pos):
    """Compute pairwise distance matrix for n points. pos is (n,2)."""
    # Use broadcasting: (n,1,2) - (1,n,2) -> (n,n,2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _max_radius_at(pos, dist_mat):
    """For each circle, compute the max radius it can have given current positions.
    Returns array of max radii (n,)."""
    n = len(pos)
    # Distance to boundaries
    r_bound = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                         np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    # Distance to other circles: r_i <= (d_ij - r_j) / 2 for each pair
    # But we want max r_i such that for all j: r_i + r_j <= d_ij
    # With fixed r_j, max r_i = min_j (d_ij - r_j)
    # But r_j is also variable. For the optimization, we use the constraint:
    # r_i + r_j <= d_ij for all i != j
    # r_i <= d_ij - r_j
    # So max r_i given other r's: min_j (d_ij - r_j)
    # But this is circular. Instead, we optimize directly.
    return r_bound


def _objective_and_grad(params):
    """
    params: (n,3) array of (x, y, r)
    Objective: maximize sum(r) subject to:
      - r_i + r_j <= d_ij for all i<j
      - r_i <= x_i, 1-x_i, y_i, 1-y_i
    We use a penalty approach: minimize -sum(r) + penalty * sum(max(0, violation)^2)
    
    Actually, let's use a cleaner approach:
    Maximize sum(r_i) subject to r_i + r_j <= d_ij and r_i <= boundary_dist_i
    
    We'll use L-BFGS-B with explicit bounds and constraints handled via the objective.
    
    Better: parameterize so constraints are automatically satisfied where possible,
    and use a smooth penalty for the pairwise constraints.
    """
    n = params.shape[0]
    pos = params[:, :2]
    r = params[:, 2]
    
    # Ensure r >= 0
    r_eff = np.maximum(r, 0)
    
    # Sum of radii
    sum_r = np.sum(r_eff)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    r_bound = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                         np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    
    # Penalty for boundary violations
    bound_viol = np.maximum(0, r_eff - r_bound)
    
    # Penalty for pairwise violations: r_i + r_j - d_ij > 0
    # For i < j
    iu = np.triu_indices(n, k=1)
    pair_viol = r_eff[iu[0]] + r_eff[iu[1]] - dist[iu]
    pair_viol = np.maximum(0, pair_viol)
    
    # Smooth penalty (soft)
    penalty_weight = 100.0
    penalty = penalty_weight * (np.sum(bound_viol ** 2) + np.sum(pair_viol ** 2))
    
    return -sum_r + penalty


def _objective_and_grad_smooth(params, beta=50.0):
    """
    Smoother objective using log-sum-exp style soft constraints.
    """
    n = params.shape[0]
    pos = params[:, :2]
    r = np.maximum(params[:, 2], 0)
    
    sum_r = np.sum(r)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # For each circle i, the effective max radius is:
    # min(boundary_dist_i, min_j (dist_ij - r_j))
    # But we want to maximize sum of r's.
    
    # Use a different approach: directly optimize with constraints
    # f = -sum(r) + penalty
    
    r_bound = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                         np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    
    # Soft boundary penalty
    bound_viol = np.maximum(0, r - r_bound)
    soft_bound = np.sum(np.logaddexp(0, beta * bound_viol) / beta)  # smooth max
    
    # Soft pairwise penalty
    iu = np.triu_indices(n, k=1)
    pair_viol = r[iu[0]] + r[iu[1]] - dist[iu]
    soft_pair = np.sum(np.logaddexp(0, beta * pair_viol) / beta)
    
    penalty_weight = 50.0
    obj = -sum_r + penalty_weight * (soft_bound + soft_pair)
    
    return obj


def _gradient(params, beta=50.0):
    """Gradient of the smooth objective."""
    n = params.shape[0]
    pos = params[:, :2]
    r_raw = params[:, 2]
    r = np.maximum(r_raw, 0)
    
    grad = np.zeros_like(params)
    
    r_bound = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                         np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    
    # Gradient of -sum(r)
    grad[:, 2] = -1.0 * (r_raw > 0).astype(float)
    
    # Boundary penalty gradient
    bound_viol = np.maximum(0, r - r_bound)
    # d/dx of logaddexp(0, beta*v)/beta where v = max(0, r - r_bound)
    # = sigmoid(beta*v) * beta * dv/dx / beta = sigmoid(beta*v) * dv/dx
    sig_bound = 1.0 / (1.0 + np.exp(-beta * bound_viol))
    # dv/dx for each dimension
    # r_bound = min(x, 1-x, y, 1-y)
    # dr_bound/dx = 1 if x < 1-x and x < y and x < 1-y, else -1 if 1-x is min, etc.
    # This is complex; use subgradient
    xb = pos[:, 0]
    yb = pos[:, 1]
    r_bound_x = np.minimum(np.minimum(xb, 1.0 - xb), np.minimum(yb, 1.0 - yb))
    r_bound_y = np.minimum(np.minimum(yb, 1.0 - yb), np.minimum(xb, 1.0 - xb))
    
    # Where is the active constraint for boundary?
    # Simplify: just use the direct gradient
    # d(r_bound)/dx = 1 if x is the min, -1 if 1-x is the min
    d_rbound_dx = np.where(xb <= 1.0 - xb, 1.0, -1.0)
    # But only if x or 1-x is actually the min of all four
    is_x_min = (xb <= 1.0 - xb) & (xb <= yb) & (xb <= 1.0 - yb)
    is_1mx_min = (1.0 - xb < xb) & (1.0 - xb <= yb) & (1.0 - xb <= 1.0 - yb)
    d_rbound_dx = np.where(is_x_min, 1.0, np.where(is_1mx_min, -1.0, 0.0))
    
    is_y_min = (yb <= 1.0 - yb) & (yb <= xb) & (yb <= 1.0 - xb)
    is_1my_min = (1.0 - yb < yb) & (1.0 - yb <= xb) & (1.0 - yb <= 1.0 - xb)
    d_rbound_dy = np.where(is_y_min, 1.0, np.where(is_1my_min, -1.0, 0.0))
    
    penalty_weight = 50.0
    
    # grad of soft_bound w.r.t. pos
    grad[:, 0] += penalty_weight * sig_bound * (-d_rbound_dx) * (r_raw > 0).astype(float)
    grad[:, 1] += penalty_weight * sig_bound * (-d_rbound_dy) * (r_raw > 0).astype(float)
    grad[:, 2] += penalty_weight * sig_bound * (r_raw > 0).astype(float)
    
    # Pairwise penalty gradient
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)  # avoid div by zero
    
    iu = np.triu_indices(n, k=1)
    pair_viol = r[iu[0]] + r[iu[1]] - dist[iu]
    sig_pair = 1.0 / (1.0 + np.exp(-beta * pair_viol))
    
    # d(pair_viol)/d(pos_i) = -diff_i / dist_ij where diff_i = pos_i - pos_j
    # For circle i (first index in iu):
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        if sig_pair[idx] < 1e-10:
            continue
        d_dist = diff[i, j] / dist[i, j]  # d(dist_ij)/d(pos_i)
        grad[i, 0] += penalty_weight * sig_pair[idx] * (-d_dist[0])
        grad[i, 1] += penalty_weight * sig_pair[idx] * (-d_dist[1])
        grad[j, 0] += penalty_weight * sig_pair[idx] * (d_dist[0])
        grad[j, 1] += penalty_weight * sig_pair[idx] * (d_dist[1])
        grad[i, 2] += penalty_weight * sig_pair[idx] * (r_raw[i] > 0).astype(float)
        grad[j, 2] += penalty_weight * sig_pair[idx] * (r_raw[j] > 0).astype(float)
    
    return grad


def _grid_init(n, rng):
    """Generate a grid-like initial configuration with some randomness."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            x = (i + 0.5) / side + rng.uniform(-0.05, 0.05)
            y = (j + 0.5) / side + rng.uniform(-0.05, 0.05)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            positions.append([x, y])
    
    # Compute initial radii
    pos = np.array(positions[:n])
    r = np.zeros(n)
    for i in range(n):
        r_bound = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        min_dist = np.inf
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
            min_dist = min(min_dist, d)
        r[i] = max(0.001, min(r_bound, min_dist / 2) * 0.8)
    
    return np.column_stack([pos, r])


def _optimize(params, max_time=30.0, beta=50.0):
    """Run L-BFGS-B optimization on the smooth objective."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    def obj(p):
        return _objective_and_grad_smooth(p, beta)
    
    def grad(p):
        return _gradient(p, beta)
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * N * 2 + [(0, 0.5)] * N
    
    result = minimize(obj, params, jac=grad, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    elapsed = time.time() - start_time
    return result.x, elapsed


def _extract_valid(params):
    """Extract a valid packing from optimized parameters."""
    pos = params[:, :2]
    r = np.maximum(params[:, 2], 0)
    
    # Clip to unit square
    pos = np.clip(pos, 0, 1)
    
    # Ensure boundary constraints
    r_bound = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                         np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    r = np.minimum(r, r_bound)
    
    # Enforce pairwise non-overlap by reducing radii
    # Iteratively adjust
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                if r[i] + r[j] > d:
                    # Reduce the larger one or split the difference
                    excess = r[i] + r[j] - d
                    if excess > 0:
                        # Reduce both proportionally
                        total = r[i] + r[j]
                        if total > 0:
                            factor = d / total
                            r[i] *= factor
                            r[j] *= factor
                            changed = True
        if not changed:
            break
    
    return np.column_stack([pos, r])


def construct():
    """Main entry point: pack 26 circles in unit square maximizing sum of radii."""
    total_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_params = None
    best_sum_r = 0.0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - total_start > deadline:
            break
        
        if restart == 0:
            # Grid initialization
            params = _grid_init(N, rng)
        else:
            # Perturbed grid
            side = int(np.ceil(np.sqrt(N)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    x = (i + 0.5) / side + rng.uniform(-0.1, 0.1)
                    y = (j + 0.5) / side + rng.uniform(-0.1, 0.1)
                    x = np.clip(x, 0.02, 0.98)
                    y = np.clip(y, 0.02, 0.98)
                    positions.append([x, y])
            pos = np.array(positions[:N])
            r = np.zeros(N)
            for i in range(N):
                r_bound = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
                min_dist = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    d = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                    min_dist = min(min_dist, d)
                r[i] = max(0.001, min(r_bound, min_dist / 2) * 0.7)
            params = np.column_stack([pos, r])
        
        # Optimize
        remaining = deadline - (time.time() - total_start)
        if remaining < 5.0:
            break
        
        opt_params, elapsed = _optimize(params, max_time=min(remaining * 0.8, 30.0))
        
        # Extract valid packing
        valid = _extract_valid(opt_params)
        sum_r = np.sum(valid[:, 2])
        
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_params = valid
        
        # Also try the raw optimized params (before extraction)
        raw_valid = _extract_valid(opt_params)
        raw_sum = np.sum(raw_valid[:, 2])
        if raw_sum > best_sum_r:
            best_sum_r = raw_sum
            best_params = raw_valid
    
    if best_params is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_params

# EVOLVE-BLOCK-END
