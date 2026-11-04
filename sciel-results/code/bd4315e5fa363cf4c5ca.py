"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _pairwise_distances(pos):
    """Compute pairwise distances between all pairs of circle centers."""
    # pos shape (N, 2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist

def _objective_and_grad(pos_r, eps=1e-6):
    """
    Objective: maximize sum of radii (so minimize negative sum).
    pos_r: (N, 3) array of (x, y, r)
    
    We use a smooth penalty approach:
    - Hard constraints: circles inside [0,1]^2, non-overlapping.
    - Soft: we directly optimize with L-BFGS-B using constraints via penalty.
    
    Actually, let's use a direct approach:
    Minimize -sum(r) subject to:
      - r_i >= 0
      - x_i - r_i >= 0, x_i + r_i <= 1
      - y_i - r_i >= 0, y_i + r_i <= 1
      - dist(i,j) >= r_i + r_j for all i < j
    
    We'll use a penalty method to handle the non-convex constraints.
    """
    n = len(pos_r)
    pos = pos_r[:, :2]
    r = pos_r[:, 2]
    
    # Objective: -sum(r)
    obj = -np.sum(r)
    
    # Gradient of objective
    grad = np.zeros_like(pos_r)
    grad[:, 2] = -1.0  # d(-sum r)/dr_i = -1
    
    # Penalty for boundary violations
    # x_i - r_i < 0 => penalty
    # x_i + r_i > 1 => penalty
    # y_i - r_i < 0 => penalty
    # y_i + r_i > 1 => penalty
    
    # Smooth penalty: use max(0, violation)^2 style but differentiable
    # Actually, let's just add large quadratic penalties
    
    penalty = 0.0
    penalty_grad = np.zeros_like(pos_r)
    
    # Boundary constraints
    # For each circle i:
    #   x_i >= r_i  =>  x_i - r_i >= 0
    #   1 - x_i >= r_i  =>  1 - x_i - r_i >= 0
    #   y_i >= r_i
    #   1 - y_i >= r_i
    
    # Violation amounts (positive when violated)
    viol_x_left = np.maximum(0, r - pos[:, 0])       # x_i < r_i
    viol_x_right = np.maximum(0, r - (1 - pos[:, 0]))  # x_i > 1 - r_i
    viol_y_left = np.maximum(0, r - pos[:, 1])
    viol_y_right = np.maximum(0, r - (1 - pos[:, 1]))
    
    # Smooth approximation: use softplus or just square
    # For differentiability, use: penalty += sum(viol^2) where viol = max(0, ...)
    # But max is not differentiable at 0. Use smooth max.
    
    # Let's use a different approach: direct gradient with subgradient-like handling
    # Since L-BFGS-B can handle bounds, let's set bounds properly.
    
    # Actually, let's reformulate:
    # Variables: x_i, y_i, r_i for i=0..N-1
    # Bounds: r_i in [0, 0.5], x_i in [0, 1], y_i in [0, 1]
    # Constraints (non-convex): dist(i,j) >= r_i + r_j
    
    # We'll use a penalty for the non-overlap constraint only.
    
    # Non-overlap penalty
    dist = _pairwise_distances(pos)
    # For each pair (i,j), i<j:
    #   dist(i,j) >= r_i + r_j
    # Violation: max(0, r_i + r_j - dist(i,j))
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    min_dist = dist[i_idx, j_idx]
    req_dist = r[i_idx] + r[j_idx]
    viol = np.maximum(0, req_dist - min_dist)
    
    penalty = np.sum(viol ** 2)
    obj += penalty * 100.0  # large penalty weight
    
    # Gradient of penalty
    # d(viol^2)/d(pos_i) for each pair
    # This is complex, let's use numerical gradient or analytic
    
    # Actually, let's just return the objective and let L-BFGS-B handle it numerically
    # Or better, let's use a different strategy.
    
    # New approach: Use scipy.optimize with constraints
    # But L-BFGS-B doesn't support general constraints.
    # Let's use SLSQP or trust-constr.
    
    # For simplicity and speed, let's use a penalty method with numerical gradients.
    # But that's slow.
    
    # Alternative: Directly optimize with a good initialization and use
    # the fact that for circle packing, we can use gradient descent on
    # a smooth objective.
    
    # Let's try a different formulation:
    # Minimize: -sum(r_i) + lambda * sum_{i<j} max(0, r_i + r_j - dist(i,j))^2
    # This is smooth if we use a smooth max.
    
    # Smooth max: softplus(x) = log(1 + exp(x))
    # But for large x, softplus(x) ≈ x
    # Let's use: smooth_max(0, x) = 0.5 * (x + sqrt(x^2 + eps^2))
    
    # This is getting complex. Let me try a simpler approach:
    # 1. Initialize with a grid
    # 2. Use gradient descent with momentum
    # 3. Project onto feasible set
    
    # Actually, let's just return the objective and let the optimizer handle it.
    # For the gradient, we'll compute it analytically for the smooth part.
    
    # The issue is the max(0, ...) is not differentiable.
    # Let's use a smooth approximation.
    
    return obj, grad

def _smooth_objective(pos_r, beta=100.0):
    """
    Smooth objective function for circle packing.
    
    pos_r: (N, 3) array of (x, y, r)
    
    We minimize: -sum(r) + (beta/2) * sum_{i<j} (max(0, r_i + r_j - dist(i,j)))^2
    + boundary penalties
    """
    n = len(pos_r)
    pos = pos_r[:, :2]
    r = pos_r[:, 2]
    
    # Base objective: -sum(r)
    obj = -np.sum(r)
    
    # Non-overlap penalty (smooth)
    dist = _pairwise_distances(pos)
    i_idx, j_idx = np.triu_indices(n, k=1)
    min_dist = dist[i_idx, j_idx]
    req_dist = r[i_idx] + r[j_idx]
    diff = req_dist - min_dist
    # Smooth max: use softplus or quadratic
    # For simplicity, use: penalty = sum(max(0, diff)^2)
    # But max is not smooth. Use: 0.5*(diff + sqrt(diff^2 + eps^2))^2 for diff > -eps
    # Simpler: just use diff^2 for all, but that penalizes even when not violated.
    # Better: use smooth_max(0, diff)^2
    
    # Let's use: smooth_max(0, x) ≈ 0.5 * (x + sqrt(x^2 + 1e-4))
    # Then square it.
    
    eps = 1e-4
    smooth_max = 0.5 * (diff + np.sqrt(diff ** 2 + eps ** 2))
    smooth_max = np.maximum(0, smooth_max)  # clip small negatives
    overlap_penalty = np.sum(smooth_max ** 2)
    
    obj += beta * overlap_penalty
    
    # Boundary penalties
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    bx_left = np.maximum(0, r - pos[:, 0])
    bx_right = np.maximum(0, r - (1 - pos[:, 0]))
    by_left = np.maximum(0, r - pos[:, 1])
    by_right = np.maximum(0, r - (1 - pos[:, 1]))
    
    bpen = np.sum(bx_left ** 2) + np.sum(bx_right ** 2) + np.sum(by_left ** 2) + np.sum(by_right ** 2)
    obj += beta * bpen
    
    return obj

def _smooth_grad(pos_r, beta=100.0):
    """
    Gradient of the smooth objective.
    """
    n = len(pos_r)
    pos = pos_r[:, :2]
    r = pos_r[:, 2]
    
    grad = np.zeros_like(pos_r)
    
    # d(-sum r)/dr = -1
    grad[:, 2] = -1.0
    
    # Non-overlap penalty gradient
    dist = _pairwise_distances(pos)
    i_idx, j_idx = np.triu_indices(n, k=1)
    min_dist = dist[i_idx, j_idx]
    req_dist = r[i_idx] + r[j_idx]
    diff = req_dist - min_dist
    
    eps = 1e-4
    # smooth_max(0, x) = 0.5 * (x + sqrt(x^2 + eps^2)) for x > -eps
    # Its derivative: 0.5 * (1 + x/sqrt(x^2 + eps^2))
    
    sqrt_term = np.sqrt(diff ** 2 + eps ** 2)
    # Avoid division by zero
    sqrt_term = np.maximum(sqrt_term, eps)
    d_smooth_max = 0.5 * (1 + diff / sqrt_term)
    
    # smooth_max is clipped to 0 for very negative values, but the derivative
    # formula handles that approximately.
    
    # Penalty = sum(smooth_max^2)
    # d(penalty)/d(diff_k) = 2 * smooth_max_k * d_smooth_max_k
    
    d_pen_ddiff = 2 * np.maximum(0, 0.5 * (diff + sqrt_term)) * d_smooth_max
    
    # Now, diff_k = r_i + r_j - dist(i,j)
    # d(diff_k)/d(r_i) = 1
    # d(diff_k)/d(r_j) = 1
    # d(diff_k)/d(pos_i) = -d(dist)/d(pos_i) = (pos_i - pos_j) / dist
    # d(diff_k)/d(pos_j) = -d(dist)/d(pos_j) = (pos_j - pos_i) / dist
    
    # For each pair (i, j):
    #   grad[r_i] += beta * d_pen_ddiff * 1
    #   grad[r_j] += beta * d_pen_ddiff * 1
    #   grad[pos_i] += beta * d_pen_ddiff * (pos_i - pos_j) / dist
    #   grad[pos_j] += beta * d_pen_ddiff * (pos_j - pos_i) / dist
    
    # Vectorized accumulation
    # For positions
    # pos_i - pos_j for each pair
    pos_diff = pos[i_idx] - pos[j_idx]
    # dist for each pair
    pair_dist = min_dist
    pair_dist = np.maximum(pair_dist, 1e-10)
    
    # Unit vector from j to i
    unit_vec = pos_diff / pair_dist[:, np.newaxis]
    
    # Gradient contribution to pos_i
    # grad[pos_i] += beta * d_pen_ddiff * unit_vec
    # Gradient contribution to pos_j
    # grad[pos_j] += beta * d_pen_ddiff * (-unit_vec)
    
    # Use np.add.at for accumulation
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        dk = d_pen_ddiff[k]
        grad[i, 2] += beta * dk  # r_i
        grad[j, 2] += beta * dk  # r_j
        grad[i, :2] += beta * dk * unit_vec[k]
        grad[j, :2] -= beta * dk * unit_vec[k]
    
    # Boundary penalties
    # bx_left = max(0, r - x)
    # d(bx_left^2)/dx = -2 * bx_left (when bx_left > 0)
    # d(bx_left^2)/dr = 2 * bx_left
    
    bx_left = np.maximum(0, r - pos[:, 0])
    bx_right = np.maximum(0, r - (1 - pos[:, 0]))
    by_left = np.maximum(0, r - pos[:, 1])
    by_right = np.maximum(0, r - (1 - pos[:, 1]))
    
    # For bx_left: violation when r > x
    # d/dx (bx_left^2) = -2 * bx_left (if bx_left > 0)
    # d/dr (bx_left^2) = 2 * bx_left
    
    mask_bl = bx_left > 0
    mask_br = bx_right > 0
    mask_yl = by_left > 0
    mask_yr = by_right > 0
    
    grad[:, 0] -= 2 * beta * bx_left  # d/dx of bx_left^2
    grad[:, 0] += 2 * beta * bx_right  # d/dx of bx_right^2 (bx_right = max(0, r - 1 + x))
    # Wait, bx_right = max(0, r - (1 - x)) = max(0, r - 1 + x)
    # d/dx (bx_right^2) = 2 * bx_right (if bx_right > 0)
    # d/dr (bx_right^2) = 2 * bx_right
    
    grad[:, 1] -= 2 * beta * by_left
    grad[:, 1] += 2 * beta * by_right
    
    grad[:, 2] += 2 * beta * bx_left
    grad[:, 2] += 2 * beta * bx_right
    grad[:, 2] += 2 * beta * by_left
    grad[:, 2] += 2 * beta * by_right
    
    return grad

def _is_valid(pos_r, tol=1e-6):
    """Check if the packing is valid (all circles inside unit square, non-overlapping)."""
    n = len(pos_r)
    pos = pos_r[:, :2]
    r = pos_r[:, 2]
    
    # Check radii non-negative
    if np.any(r < -tol):
        return False
    
    # Check inside unit square
    if np.any(pos[:, 0] - r < -tol) or np.any(pos[:, 0] + r > 1 + tol):
        return False
    if np.any(pos[:, 1] - r < -tol) or np.any(pos[:, 1] + r > 1 + tol):
        return False
    
    # Check non-overlap
    dist = _pairwise_distances(pos)
    i_idx, j_idx = np.triu_indices(n, k=1)
    min_dist = dist[i_idx, j_idx]
    req_dist = r[i_idx] + r[j_idx]
    if np.any(min_dist < req_dist - tol):
        return False
    
    return True

def _project_to_feasible(pos_r, max_iter=1000, tol=1e-8):
    """
    Project a configuration to the feasible set using iterative projection.
    This is a heuristic.
    """
    n = len(pos_r)
    pos_r = pos_r.copy()
    
    for _ in range(max_iter):
        # Clip radii to [0, 0.5]
        pos_r[:, 2] = np.clip(pos_r[:, 2], 0, 0.5)
        
        # Clip positions to [0, 1]
        pos_r[:, :2] = np.clip(pos_r[:, :2], 0, 1)
        
        # Adjust for boundary: ensure x_i >= r_i, x_i <= 1 - r_i, etc.
        # This is a projection onto the boundary constraints.
        for i in range(n):
            r = pos_r[i, 2]
            # x constraint
            if pos_r[i, 0] < r:
                pos_r[i, 0] = r
            if pos_r[i, 0] > 1 - r:
                pos_r[i, 0] = 1 - r
            # y constraint
            if pos_r[i, 1] < r:
                pos_r[i, 1] = r
            if pos_r[i, 1] > 1 - r:
                pos_r[i, 1] = 1 - r
        
        # Check non-overlap and adjust
        # If overlapping, reduce radii or move apart
        # Simple heuristic: reduce radii of overlapping pairs
        changed = False
        dist = _pairwise_distances(pos_r[:, :2])
        i_idx, j_idx = np.triu_indices(n, k=1)
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            d = dist[i, j]
            req = pos_r[i, 2] + pos_r[j, 2]
            if d < req - 1e-10:
                # Reduce radii
                overlap = req - d
                # Split the reduction
                reduce_i = overlap / 2
                reduce_j = overlap / 2
                pos_r[i, 2] = max(0, pos_r[i, 2] - reduce_i)
                pos_r[j, 2] = max(0, pos_r[j, 2] - reduce_j)
                changed = True
        
        if not changed:
            break
    
    return pos_r

def _grid_init(n):
    """Create a grid initialization for n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells[:n])

def _random_init(n, seed=42):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    r = 0.05 + 0.1 * rng.rand(n)
    x = r + (1 - 2 * r) * rng.rand(n)
    y = r + (1 - 2 * r) * rng.rand(n)
    return np.column_stack([x, y, r])

def _optimize_from_init(pos_r_init, time_limit, beta_start=10.0, beta_end=1000.0):
    """
    Optimize from a given initialization using L-BFGS-B.
    """
    n = len(pos_r_init)
    start_time = time.time()
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    # Flatten
    x0 = pos_r_init.flatten()
    
    def obj_func(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        pos_r = x.reshape(n, 3)
        return _smooth_objective(pos_r, beta=beta_start)
    
    def grad_func(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        pos_r = x.reshape(n, 3)
        return _smooth_grad(pos_r, beta=beta_start)
    
    # Run L-BFGS-B
    try:
        result = minimize(
            obj_func,
            x0,
            jac=grad_func,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        pos_r_opt = result.x.reshape(n, 3)
    except TimeoutError:
        pos_r_opt = pos_r_init
    except Exception:
        pos_r_opt = pos_r_init
    
    # Project to feasible
    pos_r_opt = _project_to_feasible(pos_r_opt)
    
    # Increase beta and re-optimize
    for beta in [beta_start * 10, beta_start * 100, beta_end]:
        if time.time() - start_time > time_limit:
            break
        
        def obj_func_b(x, b=beta):
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            pos_r = x.reshape(n, 3)
            return _smooth_objective(pos_r, beta=b)
        
        def grad_func_b(x, b=beta):
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            pos_r = x.reshape(n, 3)
            return _smooth_grad(pos_r, beta=b)
        
        try:
            result = minimize(
                obj_func_b,
                pos_r_opt.flatten(),
                jac=grad_func_b,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            pos_r_opt = result.x.reshape(n, 3)
        except TimeoutError:
            break
        except Exception:
            break
        
        pos_r_opt = _project_to_feasible(pos_r_opt)
    
    return pos_r_opt

def construct():
    start_time = time.time()
    remaining = TIME_LIMIT - (time.time() - start_time)
    
    if remaining < 5:
        # Not enough time, return grid
        return _grid_init(N)
    
    # Try multiple initializations
    inits = []
    
    # Grid init
    inits.append(_grid_init(N))
    
    # Random inits
    for seed in [42, 123, 456, 789, 1024]:
        inits.append(_random_init(N, seed=seed))
    
    # Hexagonal-like init
    # 26 circles: try a hexagonal arrangement
    # 4-5-4-5-4-4 or similar
    hex_init = []
    # Row 0: 4 circles
    for i in range(4):
        hex_init.append([(i + 0.5) / 4, 0.5 / 4, 0.5 / 4 - 1e-3])
    # Row 1: 5 circles
    for i in range(5):
        hex_init.append([(i + 0.5) / 5, 1.5 / 4, 0.5 / 4 - 1e-3])
    # Row 2: 4 circles
    for i in range(4):
        hex_init.append([(i + 0.5) / 4, 2.5 / 4, 0.5 / 4 - 1e-3])
    # Row 3: 5 circles
    for i in range(5):
        hex_init.append([(i + 0.5) / 5, 3.5 / 4, 0.5 / 4 - 1e-3])
    # Row 4: 4 circles
    for i in range(4):
        hex_init.append([(i + 0.5) / 4, 4.5 / 4, 0.5 / 4 - 1e-3])
    # Row 5: 4 circles
    for i in range(4):
        hex_init.append([(i + 0.5) / 4, 5.5 / 4, 0.5 / 4 - 1e-3])
    # That's 4+5+4+5+4+4 = 26
    hex_init = np.array(hex_init[:N])
    # Scale to fit in [0,1]^2
    # The y values go from 0.125 to 1.375, need to scale
    # Let's just use it as is and let the optimizer fix it
    inits.append(hex_init)
    
    best_score = -np.inf
    best_pos_r = None
    
    for init in inits:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Ensure valid initial radii
        init = init.copy()
        init[:, 2] = np.clip(init[:, 2], 0, 0.5)
        init[:, :2] = np.clip(init[:, :2], 0, 1)
        
        pos_r_opt = _optimize_from_init(init, TIME_LIMIT - (time.time() - start_time))
        
        # Check validity
        if _is_valid(pos_r_opt):
            score = np.sum(pos_r_opt[:, 2])
            if score > best_score:
                best_score = score
                best_pos_r = pos_r_opt
    
    if best_pos_r is None:
        # Return grid as fallback
        return _grid_init(N)
    
    return best_pos_r

# EVOLVE-BLOCK-END
