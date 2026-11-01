"""Circle packing in unit square n=32 with L-BFGS-B local search."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32
DEADLINE = 150.0  # seconds, evaluator kills at 150s
SAFETY_MARGIN = 5.0  # return at least this many seconds before deadline


def _compute_objective_and_grad(params):
    """
    Objective: maximize sum of radii, subject to:
      - circles inside [0,1]^2: x_i >= r_i, y_i >= r_i, x_i + r_i <= 1, y_i + r_i <= 1
      - non-overlap: ||p_i - p_j|| >= r_i + r_j for all i < j
    
    We minimize -sum(r_i) with soft constraints via penalty method.
    
    params: array of shape (3N,) = [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: -sum(r)
    obj = -np.sum(r)
    
    # Gradient of objective
    grad = np.zeros(3 * n)
    grad[2::3] = -1.0
    
    # Penalty for boundary violations
    # x_i >= r_i  =>  x_i - r_i >= 0
    # y_i >= r_i  =>  y_i - r_i >= 0
    # x_i + r_i <= 1  =>  1 - x_i - r_i >= 0
    # y_i + r_i <= 1  =>  1 - y_i - r_i >= 0
    # r_i > 0
    
    penalty = 0.0
    mu = 1000.0  # penalty weight
    
    # Boundary: x_i - r_i >= 0
    viol_x_left = np.maximum(0, r - x)
    penalty += mu * np.sum(viol_x_left ** 2)
    # grad contribution: d/dx_i = -2*mu*viol, d/dr_i = 2*mu*viol
    idx = np.arange(n)
    g_x = np.zeros(n)
    g_r = np.zeros(n)
    mask = viol_x_left > 0
    g_x[mask] = -2 * mu * viol_x_left[mask]
    g_r[mask] = 2 * mu * viol_x_left[mask]
    grad[0::3] += g_x
    grad[2::3] += g_r
    
    # Boundary: y_i - r_i >= 0
    viol_y_left = np.maximum(0, r - y)
    penalty += mu * np.sum(viol_y_left ** 2)
    g_y = np.zeros(n)
    g_r2 = np.zeros(n)
    mask = viol_y_left > 0
    g_y[mask] = -2 * mu * viol_y_left[mask]
    g_r2[mask] = 2 * mu * viol_y_left[mask]
    grad[1::3] += g_y
    grad[2::3] += g_r2
    
    # Boundary: 1 - x_i - r_i >= 0
    viol_x_right = np.maximum(0, x + r - 1)
    penalty += mu * np.sum(viol_x_right ** 2)
    g_x2 = np.zeros(n)
    g_r3 = np.zeros(n)
    mask = viol_x_right > 0
    g_x2[mask] = 2 * mu * viol_x_right[mask]
    g_r3[mask] = 2 * mu * viol_x_right[mask]
    grad[0::3] += g_x2
    grad[2::3] += g_r3
    
    # Boundary: 1 - y_i - r_i >= 0
    viol_y_right = np.maximum(0, y + r - 1)
    penalty += mu * np.sum(viol_y_right ** 2)
    g_y2 = np.zeros(n)
    g_r4 = np.zeros(n)
    mask = viol_y_right > 0
    g_y2[mask] = 2 * mu * viol_y_right[mask]
    g_r4[mask] = 2 * mu * viol_y_right[mask]
    grad[1::3] += g_y2
    grad[2::3] += g_r4
    
    # Non-overlap: ||p_i - p_j|| >= r_i + r_j
    # Soft penalty: max(0, (r_i + r_j) - dist_ij)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                viol = min_dist - dist
                penalty += mu * viol ** 2
                # Gradient: d/dx_i = -2*mu*viol * (dx/dist), etc.
                # d(viol)/dx_i = d(min_dist - dist)/dx_i = -dx/dist
                # d(viol)/dr_i = 1
                if dist > 1e-10:
                    factor = 2 * mu * viol / dist
                    grad[3 * i + 0] += -factor * dx
                    grad[3 * i + 1] += -factor * dy
                    grad[3 * j + 0] += factor * dx
                    grad[3 * j + 1] += factor * dy
                grad[3 * i + 2] += 2 * mu * viol
                grad[3 * j + 2] += 2 * mu * viol
    
    # Radius positivity: r_i >= 0
    viol_r = np.maximum(0, -r)
    penalty += mu * np.sum(viol_r ** 2)
    g_r5 = np.zeros(n)
    mask = viol_r > 0
    g_r5[mask] = -2 * mu * viol_r[mask]
    grad[2::3] += g_r5
    
    total_obj = obj + penalty
    
    return total_obj, grad


def _is_valid(circles, tol=1e-6):
    """Check if circles are valid (inside square, non-overlapping)."""
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check inside square
    if np.any(x < -tol) or np.any(y < -tol) or np.any(x > 1 + tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    # Check that circles don't extend outside
    if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol) or np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _make_hex_initialization(seed=0):
    """Create a hexagonal-like initial configuration."""
    rng = np.random.RandomState(seed)
    n = N
    # Start with a grid and add some jitter
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            # Hexagonal offset for odd rows
            offset = 0.5 / side if i % 2 == 1 else 0
            x = (i + 0.5 + offset) / side
            y = (j + 0.5) / side
            positions.append((x, y))
    
    # Add random jitter
    jitter_scale = 0.02
    params = np.zeros(3 * n)
    for k in range(n):
        params[3 * k + 0] = positions[k][0] + rng.randn() * jitter_scale
        params[3 * k + 1] = positions[k][1] + rng.randn() * jitter_scale
        params[3 * k + 2] = 0.5 / side * 0.8  # initial radius
    
    return params


def _make_random_initialization(seed=0):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    n = N
    params = np.zeros(3 * n)
    for k in range(n):
        params[3 * k + 0] = rng.uniform(0.1, 0.9)
        params[3 * k + 1] = rng.uniform(0.1, 0.9)
        params[3 * k + 2] = rng.uniform(0.02, 0.08)
    return params


def _local_search(params, max_iter=200, time_limit=10.0):
    """Run L-BFGS-B local search from given params."""
    start_time = time.time()
    
    def obj_and_grad(p):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise Exception("Time limit exceeded")
        return _compute_objective_and_grad(p)
    
    try:
        result = minimize(
            obj_and_grad,
            params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 2}
        )
        return result.x, result.fun
    except Exception as e:
        return params, _compute_objective_and_grad(params)[0]


def construct():
    """Construct 32 disjoint circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple initializations and local searches
    num_restarts = 20
    
    for restart in range(num_restarts):
        # Check time
        if time.time() > deadline:
            break
        
        # Create initial params
        if restart < 5:
            params = _make_hex_initialization(seed=restart)
        elif restart < 10:
            params = _make_random_initialization(seed=restart)
        else:
            # Perturb best solution found so far
            if best_params is not None:
                rng = np.random.RandomState(restart * 1000)
                params = best_params + rng.randn(3 * N) * 0.05
            else:
                params = _make_hex_initialization(seed=restart + 100)
        
        # Clip to reasonable bounds
        params[0::3] = np.clip(params[0::3], 0.01, 0.99)  # x
        params[1::3] = np.clip(params[1::3], 0.01, 0.99)  # y
        params[2::3] = np.clip(params[2::3], 0.005, 0.2)  # r
        
        # Run local search
        time_per_restart = max(1.0, (deadline - time.time()) / (num_restarts - restart))
        result_params, result_score = _local_search(params, max_iter=150, time_limit=time_per_restart)
        
        # Extract circles
        x = result_params[0::3]
        y = result_params[1::3]
        r = result_params[2::3]
        
        # Check validity
        circles = np.column_stack([x, y, r])
        if _is_valid(circles):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_params = result_params.copy()
    
    # If no valid solution found, use a simple valid fallback
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)
    
    # Final polish: run one more local search with tighter settings
    if time.time() < deadline - 2.0:
        result_params, result_score = _local_search(best_params, max_iter=100, time_limit=deadline - time.time() - 1.0)
        x = result_params[0::3]
        y = result_params[1::3]
        r = result_params[2::3]
        circles = np.column_stack([x, y, r])
        if _is_valid(circles) and np.sum(r) > best_score:
            best_params = result_params
    
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    return np.column_stack([x, y, r])
