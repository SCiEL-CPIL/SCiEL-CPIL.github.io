"""Improved circle packing for n=32 in unit square using L-BFGS-B."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 32
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _objective_and_grad(vars, penalty_weight=1000.0):
    """
    Objective function and gradient for minimizing -sum(r) + penalty.
    vars: array of shape (3*N,) -> [x1, y1, r1, x2, y2, r2, ...]
    """
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    obj_val = -np.sum(r)
    grad = np.zeros_like(vars)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # 1. Boundary Penalties
    # Constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # Penalty: sum (max(0, violation)^2)
    
    # Left: x < r -> viol = r - x
    viol_left = np.maximum(0, r - x)
    obj_val += penalty_weight * np.sum(viol_left**2)
    mask = viol_left > 0
    if np.any(mask):
        # d/dx (w * (r-x)^2) = w * 2(r-x) * (-1)
        # d/dr (w * (r-x)^2) = w * 2(r-x) * (1)
        grad[0::3][mask] += penalty_weight * 2 * viol_left[mask] * (-1)
        grad[2::3][mask] += penalty_weight * 2 * viol_left[mask] * (1)
        
    # Right: x > 1-r -> viol = x + r - 1
    viol_right = np.maximum(0, x + r - 1)
    obj_val += penalty_weight * np.sum(viol_right**2)
    mask = viol_right > 0
    if np.any(mask):
        # d/dx (w * (x+r-1)^2) = w * 2(x+r-1) * (1)
        # d/dr (w * (x+r-1)^2) = w * 2(x+r-1) * (1)
        grad[0::3][mask] += penalty_weight * 2 * viol_right[mask] * (1)
        grad[2::3][mask] += penalty_weight * 2 * viol_right[mask] * (1)
        
    # Bottom: y < r -> viol = r - y
    viol_bottom = np.maximum(0, r - y)
    obj_val += penalty_weight * np.sum(viol_bottom**2)
    mask = viol_bottom > 0
    if np.any(mask):
        grad[1::3][mask] += penalty_weight * 2 * viol_bottom[mask] * (-1)
        grad[2::3][mask] += penalty_weight * 2 * viol_bottom[mask] * (1)
        
    # Top: y > 1-r -> viol = y + r - 1
    viol_top = np.maximum(0, y + r - 1)
    obj_val += penalty_weight * np.sum(viol_top**2)
    mask = viol_top > 0
    if np.any(mask):
        grad[1::3][mask] += penalty_weight * 2 * viol_top[mask] * (1)
        grad[2::3][mask] += penalty_weight * 2 * viol_top[mask] * (1)

    # 2. Overlap Penalties
    # For each pair i < j, if dist < ri + rj, add penalty (ri + rj - dist)^2
    # Vectorized computation for pairs is tricky due to the conditional, 
    # but N=32 is small enough for a loop or block-wise vectorization.
    # We will use a loop for clarity and correctness, which is fast for N=32.
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)
            min_dist = r[i] + r[j]
            
            if dist < min_dist:
                viol = min_dist - dist
                obj_val += penalty_weight * viol**2
                
                if dist > 1e-10:
                    inv_dist = 1.0 / dist
                    # d(dist)/dx_i = -dx/dist, d(dist)/dx_j = dx/dist
                    # d(viol)/dx_i = -d(dist)/dx_i = dx/dist
                    # d(viol)/dx_j = -d(dist)/dx_j = -dx/dist
                    # d(viol)/dr_i = 1, d(viol)/dr_j = 1
                    
                    coeff = penalty_weight * 2 * viol * inv_dist
                    
                    # Gradient for i
                    grad[3*i + 0] += coeff * dx
                    grad[3*i + 1] += coeff * dy
                    grad[3*i + 2] += penalty_weight * 2 * viol
                    
                    # Gradient for j
                    grad[3*j + 0] -= coeff * dx
                    grad[3*j + 1] -= coeff * dy
                    grad[3*j + 2] += penalty_weight * 2 * viol
                else:
                    # Degenerate case: centers coincide
                    # Push them apart arbitrarily
                    grad[3*i + 0] += penalty_weight * 2 * viol
                    grad[3*j + 0] -= penalty_weight * 2 * viol
                    grad[3*i + 2] += penalty_weight * 2 * viol
                    grad[3*j + 2] += penalty_weight * 2 * viol

    return obj_val, grad

def _is_valid(vars, tol=1e-6):
    """Check if the configuration is valid."""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary check
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    if np.any(r <= 0):
        return False
        
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _hex_init():
    """Hexagonal grid initialization."""
    # Try to fit a hex pattern. 
    # For 32 circles, a 6x6 grid has 36 spots. 
    # We can use a subset or adjust spacing.
    # Let's create a grid of points and pick 32.
    points = []
    # Hex spacing: dx = 2r, dy = sqrt(3)r
    # If we assume r ~ 0.12, dx=0.24, dy=0.207
    # Width 1.0 / 0.24 ~ 4.16 -> 5 columns
    # Height 1.0 / 0.207 ~ 4.8 -> 5 rows
    # 5x5 = 25. Not enough.
    # Try tighter packing or more rows.
    # Let's just generate a dense hex grid and select 32 closest to center.
    
    r_est = 0.1
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    for row in range(10):
        y = 0.5 + row * dy
        if y > 1.0: break
        offset = dx / 2 if row % 2 == 1 else 0
        for col in range(10):
            x = 0.5 + col * dx - 5*dx + offset
            if x > 1.0: break
            if x < 0: continue
            points.append([x, y])
            
    points = np.array(points)
    # Select 32 closest to center (0.5, 0.5)
    dists = np.sum((points - 0.5)**2, axis=1)
    idx = np.argsort(dists)[:N]
    init_points = points[idx]
    
    vars = np.zeros(3*N)
    vars[0::3] = init_points[:, 0]
    vars[1::3] = init_points[:, 1]
    vars[2::3] = 0.05 # Small initial radius
    
    return vars

def _random_init():
    """Random initialization."""
    vars = np.zeros(3*N)
    # Spread them out roughly
    x = np.linspace(0.1, 0.9, N) + np.random.uniform(-0.05, 0.05, N)
    y = np.linspace(0.1, 0.9, N) + np.random.uniform(-0.05, 0.05, N)
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    
    vars[0::3] = x
    vars[1::3] = y
    vars[2::3] = 0.05
    return vars

def _grid_init():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9
                cells.append((x, y, r))
    
    vars = np.zeros(3*N)
    for i, (x, y, r) in enumerate(cells):
        vars[3*i] = x
        vars[3*i+1] = y
        vars[3*i+2] = r
    return vars

def construct():
    start_time = time.time()
    
    best_vars = None
    best_score = -np.inf
    
    # Different initialization strategies
    inits = [
        _hex_init(),
        _grid_init(),
        _random_init(),
        _random_init()
    ]
    
    # Run optimization for each init
    for init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # L-BFGS-B Optimization
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # We use a high penalty weight to enforce constraints strongly
        # but L-BFGS-B also respects the bounds.
        # The penalty handles overlaps.
        
        try:
            res = minimize(
                _objective_and_grad,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 20}
            )
        except Exception as e:
            # If optimization fails, skip this init
            continue
            
        if time.time() - start_time > DEADLINE - 2:
            break
            
        final_vars = res.x
        
        # Check validity
        if _is_valid(final_vars):
            score = np.sum(final_vars[2::3])
            if score > best_score:
                best_score = score
                best_vars = final_vars.copy()
                
    # If we found a valid solution, try to polish it
    if best_vars is not None:
        # Polish: Try to increase radii slightly by running another optimization
        # with a lower penalty weight, focusing more on the objective -sum(r)
        # This helps if the previous run was too conservative due to high penalty.
        
        # However, if the solution is strictly valid, we can try a local search
        # or a second L-BFGS-B pass with tuned parameters.
        
        # Let's try a second pass with a slightly lower penalty weight to allow
        # for larger radii if there is slack.
        if time.time() - start_time < DEADLINE - 10:
            try:
                res2 = minimize(
                    lambda v: _objective_and_grad(v, penalty_weight=100.0)[0],
                    best_vars,
                    jac=lambda v: _objective_and_grad(v, penalty_weight=100.0)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
                )
                if _is_valid(res2.x):
                    score2 = np.sum(res2.x[2::3])
                    if score2 > best_score:
                        best_score = score2
                        best_vars = res2.x.copy()
            except Exception:
                pass
    
    # Final check and formatting
    if best_vars is not None:
        result = np.zeros((N, 3))
        result[:, 0] = best_vars[0::3]
        result[:, 1] = best_vars[1::3]
        result[:, 2] = best_vars[2::3]
        return result
    else:
        # Fallback to a known valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
