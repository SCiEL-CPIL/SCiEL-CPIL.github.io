"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _objective(params, w, h):
    """
    Objective: maximize sum of radii, with penalty for constraint violations.
    params: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = N
    xs = params[0:3*n:3]
    ys = params[1:3*n:3]
    rs = params[2:3*n:3]
    
    # Penalty for negative radii
    penalty = 0.0
    penalty += 100.0 * np.sum(np.maximum(0, -rs)) ** 2
    
    # Wall constraints
    for i in range(n):
        # Left wall: x_i >= r_i
        viol = max(0, rs[i] - xs[i])
        penalty += 1000.0 * viol ** 2
        # Right wall: x_i <= w - r_i
        viol = max(0, xs[i] - (w - rs[i]))
        penalty += 1000.0 * viol ** 2
        # Bottom wall: y_i >= r_i
        viol = max(0, rs[i] - ys[i])
        penalty += 1000.0 * viol ** 2
        # Top wall: y_i <= h - r_i
        viol = max(0, ys[i] - (h - rs[i]))
        penalty += 1000.0 * viol ** 2
    
    # Circle-circle constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += 1000.0 * overlap ** 2
    
    # We want to maximize sum of radii, so minimize negative sum
    return -np.sum(rs) + penalty

def _objective_and_grad(params, w, h):
    """
    Objective with gradient for L-BFGS-B.
    """
    n = N
    xs = params[0:3*n:3]
    ys = params[1:3*n:3]
    rs = params[2:3*n:3]
    
    obj_val = -np.sum(rs)
    grad = np.zeros(3*n)
    
    # Gradient of -sum(r)
    grad[2:3*n:3] = -1.0
    
    # Penalty gradients
    pen_coeff = 1000.0
    
    # Wall constraints
    for i in range(n):
        # Left wall: x_i >= r_i => violation = max(0, r_i - x_i)
        viol = rs[i] - xs[i]
        if viol > 0:
            # d(obj)/dx_i += 2 * pen_coeff * viol * (-1)  [from -x_i in viol]
            # d(obj)/dr_i += 2 * pen_coeff * viol * (1)   [from +r_i in viol]
            grad[3*i] += -2 * pen_coeff * viol
            grad[3*i+2] += 2 * pen_coeff * viol
        
        # Right wall: x_i <= w - r_i => violation = max(0, x_i - (w - r_i))
        viol = xs[i] - (w - rs[i])
        if viol > 0:
            # d(obj)/dx_i += 2 * pen_coeff * viol * (1)
            # d(obj)/dr_i += 2 * pen_coeff * viol * (1)  [from +r_i in (w-r_i) => -(-r_i) = +r_i]
            grad[3*i] += 2 * pen_coeff * viol
            grad[3*i+2] += 2 * pen_coeff * viol
        
        # Bottom wall: y_i >= r_i
        viol = rs[i] - ys[i]
        if viol > 0:
            grad[3*i+1] += -2 * pen_coeff * viol
            grad[3*i+2] += 2 * pen_coeff * viol
        
        # Top wall: y_i <= h - r_i
        viol = ys[i] - (h - rs[i])
        if viol > 0:
            grad[3*i+1] += 2 * pen_coeff * viol
            grad[3*i+2] += 2 * pen_coeff * viol
    
    # Circle-circle constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-20 else 1e-10
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap = min_dist - dist
                # d(obj)/dx_i += 2 * pen_coeff * overlap * (dx/dist)
                # d(obj)/dx_j += 2 * pen_coeff * overlap * (-dx/dist)
                # d(obj)/dr_i += 2 * pen_coeff * overlap * 1
                # d(obj)/dr_j += 2 * pen_coeff * overlap * 1
                factor = 2 * pen_coeff * overlap / dist
                grad[3*i] += factor * dx
                grad[3*i+1] += factor * dy
                grad[3*i+2] += 2 * pen_coeff * overlap
                grad[3*j] -= factor * dx
                grad[3*j+1] -= factor * dy
                grad[3*j+2] += 2 * pen_coeff * overlap
    
    return obj_val, grad

def _evaluate_validity(xs, ys, rs, w, h, tol=1e-7):
    """Check if packing is valid. Returns True if valid."""
    for i in range(N):
        if xs[i] < rs[i] - tol or xs[i] > w - rs[i] + tol:
            return False
        if ys[i] < rs[i] - tol or ys[i] > h - rs[i] + tol:
            return False
    
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _sum_radii(xs, ys, rs):
    return float(np.sum(rs))

def _solve_for_aspect(w, h, deadline, initial_params=None):
    """
    Solve for given aspect ratio w x h (w + h = 2).
    Returns best (xs, ys, rs, sum_r) found.
    """
    n = N
    best_sum = -np.inf
    best_xs = None
    best_ys = None
    best_rs = None
    
    # Try multiple initializations
    inits = []
    
    # 1. Grid initialization
    for rows in [3, 4, 5, 6]:
        for cols in [3, 4, 5, 6, 7]:
            if rows * cols < n:
                continue
            xs_grid = []
            ys_grid = []
            for i in range(rows):
                for j in range(cols):
                    x = (j + 0.5) * w / cols
                    y = (i + 0.5) * h / rows
                    xs_grid.append(x)
                    ys_grid.append(y)
                    if len(xs_grid) >= n:
                        break
                if len(xs_grid) >= n:
                    break
            if len(xs_grid) >= n:
                xs_grid = np.array(xs_grid[:n])
                ys_grid = np.array(ys_grid[:n])
                
                # Estimate max possible radius
                min_pair_dist = np.inf
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.sqrt((xs_grid[i]-xs_grid[j])**2 + (ys_grid[i]-ys_grid[j])**2)
                        min_pair_dist = min(min_pair_dist, d)
                for i in range(n):
                    d_wall = min(xs_grid[i], w-xs_grid[i], ys_grid[i], h-ys_grid[i])
                    min_pair_dist = min(min_pair_dist, d_wall)
                
                r_est = min_pair_dist / 2.0 * 0.9
                rs_grid = np.full(n, r_est)
                
                params = np.zeros(3*n)
                params[0:3*n:3] = xs_grid
                params[1:3*n:3] = ys_grid
                params[2:3*n:3] = rs_grid
                inits.append(params)
    
    # 2. Hexagonal-like initialization
    hex_rows = 5
    hex_cols = 5
    xs_hex = []
    ys_hex = []
    for i in range(hex_rows):
        for j in range(hex_cols):
            x = (j + 0.5 + (0.5 if i % 2 == 1 else 0)) * w / hex_cols
            y = (i + 0.5) * h / hex_rows
            if 0 < x < w and 0 < y < h:
                xs_hex.append(x)
                ys_hex.append(y)
                if len(xs_hex) >= n:
                    break
        if len(xs_hex) >= n:
            break
    if len(xs_hex) >= n:
        xs_hex = np.array(xs_hex[:n])
        ys_hex = np.array(ys_hex[:n])
        min_pair_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((xs_hex[i]-xs_hex[j])**2 + (ys_hex[i]-ys_hex[j])**2)
                min_pair_dist = min(min_pair_dist, d)
        for i in range(n):
            d_wall = min(xs_hex[i], w-xs_hex[i], ys_hex[i], h-ys_hex[i])
            min_pair_dist = min(min_pair_dist, d_wall)
        r_est = min_pair_dist / 2.0 * 0.9
        rs_hex = np.full(n, r_est)
        params = np.zeros(3*n)
        params[0:3*n:3] = xs_hex
        params[1:3*n:3] = ys_hex
        params[2:3*n:3] = rs_hex
        inits.append(params)
    
    # 3. Random initialization
    np.random.seed(12345)
    for _ in range(3):
        xs_rand = np.random.uniform(0.1, w-0.1, n)
        ys_rand = np.random.uniform(0.1, h-0.1, n)
        rs_rand = np.full(n, 0.08)
        params = np.zeros(3*n)
        params[0:3*n:3] = xs_rand
        params[1:3*n:3] = ys_rand
        params[2:3*n:3] = rs_rand
        inits.append(params)
    
    # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
    r_max = min(w, h) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.0, r_max))
    
    # Run optimization from each initialization
    for init_params in inits:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_and_grad,
                init_params,
                args=(w, h),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < best_sum:
                xs = result.x[0:3*n:3]
                ys = result.x[1:3*n:3]
                rs = result.x[2:3*n:3]
                
                # Check validity
                if _evaluate_validity(xs, ys, rs, w, h):
                    s = _sum_radii(xs, ys, rs)
                    if s > best_sum:
                        best_sum = s
                        best_xs = xs.copy()
                        best_ys = ys.copy()
                        best_rs = rs.copy()
        except Exception:
            pass
    
    if best_xs is not None:
        return best_xs, best_ys, best_rs, best_sum
    return None, None, None, -np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_sum_r = 0.0
    best_result = None
    
    # Sample aspect ratios: w + h = 2, w in [0.5, 1.5]
    # By symmetry, w in [1.0, 1.5] is equivalent to h in [0.5, 1.0]
    # So we can just check w in [1.0, 1.5]
    w_values = np.linspace(1.0, 1.5, 12)
    
    for w in w_values:
        if time.time() > deadline:
            break
        
        h = 2.0 - w
        
        xs, ys, rs, s = _solve_for_aspect(w, h, deadline)
        
        if xs is not None and s > best_sum_r:
            best_sum_r = s
            best_result = np.column_stack([xs, ys, rs])
    
    if best_result is None:
        # Fallback: simple grid in a square
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (j + 0.5) * w / side
                y = (i + 0.5) * h / side
                r = 0.5 / side * 0.95
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
