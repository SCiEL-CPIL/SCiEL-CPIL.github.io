"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 26
_DEADLINE = 135.0
_START_TIME = 0.0


def _elapsed():
    return time.time() - _START_TIME


def _is_valid(params, tol=1e-7):
    """Check if configuration is valid: circles inside [0,1]^2 with disjoint interiors."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    for i in range(N):
        if x[i] < r[i] - tol or x[i] > 1.0 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1.0 - r[i] + tol:
            return False
        if r[i] < -tol:
            return False
    # Check pairwise distances
    dists = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
    for i in range(N):
        for j in range(i + 1, N):
            if dists[i, j] < r[i] + r[j] - tol:
                return False
    return True


def _score(params):
    """Return sum of radii if valid, else -inf."""
    if _is_valid(params, tol=1e-7):
        return np.sum(params[2::3])
    return -np.inf


def _soft_objective(params, beta=500.0):
    """
    Smooth objective: minimize -sum(r) + beta * (boundary_violation + overlap_violation).
    Uses smooth max-like formulation for differentiability.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    obj = -np.sum(r)
    
    # Boundary violations (smooth)
    # Left: x - r < 0 => violation = max(0, r - x)
    # Right: x + r > 1 => violation = max(0, x + r - 1)
    # Bottom: y - r < 0
    # Top: y + r > 1
    
    # Use softplus-like: t + log(1 + exp(-t))/beta approximation, but simpler:
    # Just use squared violations for smoothness
    bv = np.maximum(0, r - x) + np.maximum(0, x + r - 1) + \
         np.maximum(0, r - y) + np.maximum(0, y + r - 1)
    obj += beta * np.sum(bv ** 2)
    
    # Overlap violations
    dists = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
    for i in range(N):
        for j in range(i + 1, N):
            overlap = r[i] + r[j] - dists[i, j]
            if overlap > 0:
                obj += beta * overlap ** 2
    
    return obj


def _soft_gradient(params, beta=500.0):
    """
    Numerical gradient of the soft objective using central differences.
    Vectorized where possible.
    """
    n = len(params)
    grad = np.zeros(n)
    eps = 1e-6
    
    # Use a smaller eps for better accuracy
    for i in range(n):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += eps
        params_m[i] -= eps
        grad[i] = (_soft_objective(params_p, beta) - _soft_objective(params_m, beta)) / (2 * eps)
    
    return grad


def _project_feasible(params):
    """
    Project parameters to be feasible: ensure circles are within bounds.
    This is a simple projection; overlaps are handled by the optimizer.
    """
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    for i in range(N):
        r[i] = max(r[i], 1e-5)
        # Clamp center so circle fits in [0,1]^2
        r[i] = min(r[i], 0.5)  # max possible radius
        x[i] = np.clip(x[i], r[i], 1.0 - r[i])
        y[i] = np.clip(y[i], r[i], 1.0 - r[i])
    
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def _grow_radii(params, max_iter=50, time_budget=10.0):
    """
    Try to increase radii of all circles simultaneously using a separate optimization.
    Fix positions, optimize only radii to maximize sum(r) subject to constraints.
    """
    start = time.time()
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    def _radius_obj(r_vec, beta=200.0):
        obj = -np.sum(r_vec)
        # Boundary
        for i in range(N):
            bv = max(0, r_vec[i] - x[i]) + max(0, x[i] + r_vec[i] - 1) + \
                 max(0, r_vec[i] - y[i]) + max(0, y[i] + r_vec[i] - 1)
            obj += beta * bv ** 2
        # Overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                ov = r_vec[i] + r_vec[j] - d
                if ov > 0:
                    obj += beta * ov ** 2
        return obj
    
    def _radius_grad(r_vec, beta=200.0):
        n_r = len(r_vec)
        grad = np.zeros(n_r)
        eps = 1e-6
        for i in range(n_r):
            rp = r_vec.copy(); rp[i] += eps
            rm = r_vec.copy(); rm[i] -= eps
            grad[i] = (_radius_obj(rp, beta) - _radius_obj(rm, beta)) / (2 * eps)
        return grad
    
    bounds = [(1e-5, 0.5) for _ in range(N)]
    
    try:
        result = minimize(
            _radius_obj, r,
            method='L-BFGS-B',
            jac=_radius_grad,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() - start < time_budget:
            new_r = result.x
            # Check validity with new radii
            new_params = params.copy()
            new_params[2::3] = new_r
            if _is_valid(new_params, tol=1e-7):
                return new_params
    except Exception:
        pass
    
    return params


def _hex_packing_init():
    """Hexagonal close-packing initialization for 26 circles."""
    params = np.zeros(3 * N)
    # 5x5 grid with hex offset, take first 26
    positions = []
    spacing = 0.19
    for row in range(6):
        for col in range(6):
            x = 0.1 + col * spacing + (spacing * 0.5 if row % 2 else 0)
            y = 0.1 + row * spacing * 0.866
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append((x, y))
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    
    for i in range(N):
        if i < len(positions):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
        else:
            params[3*i] = 0.5
            params[3*i+1] = 0.5
        params[3*i+2] = 0.08  # initial radius
    
    return _project_feasible(params)


def _grid_init(seed=42):
    """Grid initialization with perturbation."""
    rng = np.random.default_rng(seed)
    side = 5
    params = np.zeros(3 * N)
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side + rng.uniform(-0.01, 0.01)
            y = (j + 0.5) / side + rng.uniform(-0.01, 0.01)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            params[3*count] = x
            params[3*count+1] = y
            params[3*count+2] = 0.09
            count += 1
    return _project_feasible(params)


def _random_init(rng):
    """Random initialization."""
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = rng.uniform(0.1, 0.9)
        params[3*i+1] = rng.uniform(0.1, 0.9)
        params[3*i+2] = rng.uniform(0.03, 0.1)
    return _project_feasible(params)


def _spiral_init():
    """Spiral initialization placing circles from center outward."""
    params = np.zeros(3 * N)
    rng = np.random.default_rng(123)
    cx, cy = 0.5, 0.5
    for i in range(N):
        if i == 0:
            x, y = cx, cy
        else:
            angle = i * 2.399963  # golden angle
            radius = 0.08 * np.sqrt(i)
            x = cx + radius * np.cos(angle) + rng.uniform(-0.02, 0.02)
            y = cy + radius * np.sin(angle) + rng.uniform(-0.02, 0.02)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = 0.08
    return _project_feasible(params)


def _optimize_full(params_init, time_budget=20.0, beta_schedule=None):
    """
    Full optimization: optimize all 3*N parameters with penalty annealing.
    Uses multiple phases with decreasing beta.
    """
    start = time.time()
    params = params_init.copy()
    
    if beta_schedule is None:
        beta_schedule = [500.0, 200.0, 50.0, 10.0]
    
    for beta in beta_schedule:
        if time.time() - start > time_budget - 2:
            break
        
        def obj(p):
            return _soft_objective(p, beta)
        
        def grad(p):
            return _soft_gradient(p, beta)
        
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))  # x
            bounds.append((0.01, 0.99))  # y
            bounds.append((0.01, 0.49))  # r
        
        phase_budget = min(5.0, time_budget - (time.time() - start))
        if phase_budget < 1:
            break
        
        try:
            result = minimize(
                obj, params,
                method='L-BFGS-B',
                jac=grad,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            params = result.x
        except Exception:
            pass
    
    return _project_feasible(params)


def _construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_params = None
    best_score_val = -np.inf
    
    # Multiple initialization strategies
    inits = []
    inits.append(('hex', _hex_packing_init()))
    inits.append(('grid', _grid_init(42)))
    inits.append(('grid2', _grid_init(123)))
    inits.append(('spiral', _spiral_init()))
    
    rng = np.random.default_rng(2024)
    inits.append(('rand1', _random_init(rng)))
    inits.append(('rand2', _random_init(rng)))
    
    for name, init_params in inits:
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Allocate time budget
        remaining = _DEADLINE - _elapsed() - 5
        time_budget = min(18.0, remaining * 0.5)
        if time_budget < 3:
            break
        
        # Optimize
        params_opt = _optimize_full(init_params, time_budget=time_budget)
        
        # Grow radii
        if _elapsed() < _DEADLINE - 8:
            params_opt = _grow_radii(params_opt, max_iter=30, time_budget=8.0)
        
        # Evaluate
        sc = _score(params_opt)
        if sc > best_score_val:
            best_score_val = sc
            best_params = params_opt.copy()
    
    # If no valid solution found, use a fallback
    if best_params is None or best_score_val <= 0:
        # Simple grid fallback
        best_params = _grid_init(42)
        # Shrink radii to ensure validity
        for attempt in range(10):
            if _is_valid(best_params, tol=1e-7):
                break
            best_params[2::3] *= 0.9
        if not _is_valid(best_params, tol=1e-7):
            # Last resort: tiny circles on a grid
            side = 5
            best_params = np.zeros(3 * N)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count >= N:
                        break
                    best_params[3*count] = (i + 0.5) / side
                    best_params[3*count+1] = (j + 0.5) / side
                    best_params[3*count+2] = 0.05 / side
                    count += 1
    
    # Final polish: one more grow_radii pass if time allows
    if _elapsed() < _DEADLINE - 5:
        best_params = _grow_radii(best_params, max_iter=20, time_budget=5.0)
    
    return best_params.reshape(N, 3)


def construct():
    """Entry point: returns array of shape (26, 3) with rows (x, y, radius)."""
    try:
        result = _construct()
        # Final validation
        if not _is_valid(result.flatten(), tol=1e-7):
            # Shrink radii until valid
            for _ in range(20):
                if _is_valid(result.flatten(), tol=1e-7):
                    break
                result[:, 2] *= 0.9
            if not _is_valid(result.flatten(), tol=1e-7):
                # Absolute fallback
                result = np.zeros((N, 3))
                side = 5
                count = 0
                for i in range(side):
                    for j in range(side):
                        if count >= N:
                            break
                        result[count, 0] = (i + 0.5) / side
                        result[count, 1] = (j + 0.5) / side
                        result[count, 2] = 0.04
                        count += 1
        return result
    except Exception:
        # Emergency fallback
        result = np.zeros((N, 3))
        side = 5
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                result[count, 0] = (i + 0.5) / side
                result[count, 1] = (j + 0.5) / side
                result[count, 2] = 0.04
                count += 1
        return result
