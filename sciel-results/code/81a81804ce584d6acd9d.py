"""Improved program for circle packing in unit square n=32."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint


def _pairwise_constraints(x, y, r):
    """Compute all pairwise distance constraints: dist(i,j) >= r_i + r_j."""
    n = len(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = r[:, None] + r[None, :]
    # Return slack: dist - min_dist (should be >= 0)
    return dist - min_dist


def _boundary_slacks(x, y, r):
    """Compute boundary slacks: x - r >= 0, y - r >= 0, 1-x-r >= 0, 1-y-r >= 0."""
    return np.concatenate([
        x - r,
        y - r,
        1 - x - r,
        1 - y - r
    ])


def _objective_and_grad(vars, n):
    """Objective: maximize sum(r), so minimize -sum(r)."""
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:]
    val = -np.sum(r)
    grad = np.zeros_like(vars)
    grad[2*n:] = -1.0
    return val, grad


def _constraints_and_jac(vars, n):
    """Compute constraint values and Jacobian for scipy SLSQP."""
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:]
    
    # Pairwise constraints
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
    min_dist = r[:, None] + r[None, :]
    pair_slack = dist - min_dist  # shape (n, n)
    
    # Boundary constraints
    bound_slack = np.concatenate([
        x - r,
        y - r,
        1 - x - r,
        1 - y - r
    ])
    
    all_slack = np.concatenate([pair_slack.ravel(), bound_slack])
    
    # Jacobian
    jac = np.zeros((len(all_slack), len(vars)))
    
    # Pairwise part
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            row = idx
            if dist[i, j] > 1e-8:
                dxij = x[i] - x[j]
                dyij = y[i] - y[j]
                dist_ij = dist[i, j]
                # d(dist - r_i - r_j)/dx_i
                jac[row, i] = dxij / dist_ij
                jac[row, n + i] = dyij / dist_ij
                jac[row, 2*n + i] = -1.0
                # d(dist - r_i - r_j)/dx_j
                jac[row, j] = -dxij / dist_ij
                jac[row, n + j] = -dyij / dist_ij
                jac[row, 2*n + j] = -1.0
            idx += 1
    
    # Boundary part
    bidx = idx
    # x - r >= 0
    for i in range(n):
        jac[bidx + i, i] = 1.0
        jac[bidx + i, 2*n + i] = -1.0
    # y - r >= 0
    for i in range(n):
        jac[bidx + n + i, n + i] = 1.0
        jac[bidx + n + i, 2*n + i] = -1.0
    # 1 - x - r >= 0
    for i in range(n):
        jac[bidx + 2*n + i, i] = -1.0
        jac[bidx + 2*n + i, 2*n + i] = -1.0
    # 1 - y - r >= 0
    for i in range(n):
        jac[bidx + 3*n + i, n + i] = -1.0
        jac[bidx + 3*n + i, 2*n + i] = -1.0
    
    return all_slack, jac


def _is_valid(x, y, r, tol=1e-7):
    """Check if the packing is valid."""
    if np.any(r < -tol) or np.any(x < -tol) or np.any(y < -tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
        return False
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _compute_score(x, y, r):
    """Compute the sum of radii."""
    return np.sum(r)


def _grid_init(n):
    """Create a grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                rad = 0.5 / side - 1e-3
                cells.append((cx, cy, rad))
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.array([c[2] for c in cells])
    return x, y, r


def _random_init(n, seed=42):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    # Place circles in a loose grid with some randomness
    side = int(np.ceil(np.sqrt(n)))
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    for i in range(n):
        row = i // side
        col = i % side
        base_x = (col + 0.5) / side
        base_y = (row + 0.5) / side
        # Add small random perturbation
        x[i] = base_x + rng.uniform(-0.1, 0.1) / side
        y[i] = base_y + rng.uniform(-0.1, 0.1) / side
        r[i] = 0.4 / side
    
    # Ensure within bounds
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return x, y, r


def _optimize(x, y, r, deadline, max_iter=500):
    """Run SLSQP optimization."""
    n = len(r)
    vars0 = np.concatenate([x, y, r])
    
    # Bounds: 0 <= x,y <= 1, r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
    for i in range(n):
        bounds.append((0, 1))  # y
    for i in range(n):
        bounds.append((1e-8, 0.5))  # r
    
    def obj(v):
        val, grad = _objective_and_grad(v, n)
        return val, grad
    
    # Use penalty method with inequality constraints
    # We'll use a simple penalty approach
    
    best_vars = vars0.copy()
    best_score = _compute_score(x, y, r)
    
    # Try SLSQP with constraints
    try:
        # Define inequality constraints as functions
        # For SLSQP, we need to provide constraint functions
        # This is complex, so let's use a penalty-based approach
        
        # Penalty method
        mu = 1e-2
        for _ in range(30):
            if time.time() > deadline:
                break
            
            def penalized_obj(v):
                x_v = v[:n]
                y_v = v[n:2*n]
                r_v = v[2*n:]
                
                base_obj = -np.sum(r_v)
                
                # Pairwise penalty
                dx = x_v[:, None] - x_v[None, :]
                dy = y_v[:, None] - y_v[None, :]
                dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
                min_dist = r_v[:, None] + r_v[None, :]
                violation = np.maximum(0, min_dist - dist)
                pair_pen = mu * np.sum(violation ** 2)
                
                # Boundary penalty
                bound_viol = np.maximum(0, np.concatenate([
                    -(x_v - r_v),
                    -(y_v - r_v),
                    -((1 - x_v) - r_v),
                    -((1 - y_v) - r_v)
                ]))
                bound_pen = mu * np.sum(bound_viol ** 2)
                
                return base_obj + pair_pen + bound_pen
            
            res = minimize(penalized_obj, best_vars, method='BFGS',
                          bounds=bounds, options={'maxiter': 100, 'gtol': 1e-6})
            
            if res.success or res.fun < _compute_score(best_vars[:n], best_vars[n:2*n], best_vars[2*n:]):
                best_vars = res.x
                current_score = _compute_score(best_vars[:n], best_vars[n:2*n], best_vars[2*n:])
                if current_score > best_score:
                    best_score = current_score
            
            mu *= 0.5
            if mu < 1e-8:
                break
        
        x_opt = best_vars[:n]
        y_opt = best_vars[n:2*n]
        r_opt = best_vars[2*n:]
        
        # Clean up: ensure validity
        r_opt = np.maximum(r_opt, 1e-8)
        x_opt = np.clip(x_opt, r_opt, 1 - r_opt)
        y_opt = np.clip(y_opt, r_opt, 1 - r_opt)
        
        # Fix any remaining violations by shrinking circles
        for _ in range(50):
            if time.time() > deadline:
                break
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x_opt[i] - x_opt[j]
                    dy = y_opt[i] - y_opt[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    req = r_opt[i] + r_opt[j]
                    if d < req:
                        # Shrink both circles slightly
                        excess = (req - d) / 2 + 1e-9
                        r_opt[i] = max(r_opt[i] - excess, 1e-8)
                        r_opt[j] = max(r_opt[j] - excess, 1e-8)
                        valid = False
            # Check boundaries
            for i in range(n):
                if x_opt[i] < r_opt[i]:
                    x_opt[i] = r_opt[i]
                    valid = False
                if x_opt[i] + r_opt[i] > 1:
                    x_opt[i] = 1 - r_opt[i]
                    valid = False
                if y_opt[i] < r_opt[i]:
                    y_opt[i] = r_opt[i]
                    valid = False
                if y_opt[i] + r_opt[i] > 1:
                    y_opt[i] = 1 - r_opt[i]
                    valid = False
            if valid:
                break
        
        return x_opt, y_opt, r_opt
        
    except Exception:
        return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_x, best_y, best_r = None, None, None
    best_score = 0.0
    
    # Try multiple initializations
    inits = []
    
    # Grid init
    x, y, r = _grid_init(N)
    inits.append((x.copy(), y.copy(), r.copy()))
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789]:
        if time.time() > deadline - 20:
            break
        x, y, r = _random_init(N, seed)
        inits.append((x.copy(), y.copy(), r.copy()))
    
    # Also try a more structured init: two rows of 16
    x2 = np.zeros(N)
    y2 = np.zeros(N)
    r2 = np.zeros(N)
    for i in range(N):
        row = i // 16
        col = i % 16
        x2[i] = (col + 0.5) / 16
        y2[i] = (row + 0.5) / 2
        r2[i] = 0.4 / 16
    inits.append((x2.copy(), y2.copy(), r2.copy()))
    
    for init_idx, (x0, y0, r0) in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        x_opt, y_opt, r_opt = _optimize(x0, y0, r0, deadline)
        
        score = _compute_score(x_opt, y_opt, r_opt)
        
        if _is_valid(x_opt, y_opt, r_opt):
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_opt.copy()
    
    # If we didn't find a valid solution, fall back to grid
    if best_x is None:
        x, y, r = _grid_init(N)
        best_x, best_y, best_r = x, y, r
    
    result = np.column_stack([best_x, best_y, best_r])
    return result
# EVOLVE-BLOCK-END
