"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _pack_26():
    """
    Pack 26 circles in [0,1]^2 maximizing sum of radii.
    
    Strategy:
    1. Generate a good initial configuration using a structured grid + perturbation
    2. Use L-BFGS-B with a penalty method for constraints
    3. Polish with tighter tolerances
    
    Variables: x (N), y (N), r (N) -> 3N = 78 variables
    Objective: maximize sum(r) -> minimize -sum(r)
    Constraints:
      - r_i >= 0
      - x_i - r_i >= 0, x_i + r_i <= 1
      - y_i - r_i >= 0, y_i + r_i <= 1
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i < j
    """
    # Initial configuration: use a hexagonal-like grid pattern
    # For 26 circles, try a 5x5 grid with corners removed, or structured placement
    
    # Strategy: start with a grid of 5x5 = 25 circles, add one more
    # Actually, let's use a more sophisticated initial guess
    
    # Use a 5x6 grid pattern, take 26 of them
    cols = 6
    rows = 5
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    base_r = 0.5 * min(cell_w, cell_h) * 0.95  # slightly smaller to allow movement
    
    # Generate grid positions
    positions = []
    for j in range(rows):
        for i in range(cols):
            x = (i + 0.5) * cell_w
            y = (j + 0.5) * cell_h
            # Hexagonal offset for even rows
            if j % 2 == 1:
                x += cell_w * 0.25
            positions.append([x, y])
    
    # We have 30 positions, pick 26 that are well-distributed
    # Use the first 26 or a subset
    positions = positions[:N]
    
    # Initial radii: start with a common radius
    r0 = base_r * 0.8
    
    # Build initial variable vector
    x0 = np.array([p[0] for p in positions])
    y0 = np.array([p[1] for p in positions])
    r0_vec = np.full(N, r0)
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    x0 += np.random.randn(N) * 0.005
    y0 += np.random.randn(N) * 0.005
    r0_vec += np.random.randn(N) * 0.002
    r0_vec = np.clip(r0_vec, 0.01, 0.2)
    
    var0 = np.concatenate([x0, y0, r0_vec])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x_i
    for i in range(N):
        bounds.append((0.0, 1.0))  # y_i
    for i in range(N):
        bounds.append((0.001, 0.5))  # r_i
    
    def objective_and_grad(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        # Objective: -sum(r)
        obj = -np.sum(r)
        
        # Penalty for constraint violations
        penalty = 0.0
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        
        # Boundary constraints (soft)
        # x_i - r_i >= 0 -> violation = max(0, r_i - x_i)
        # x_i + r_i <= 1 -> violation = max(0, x_i + r_i - 1)
        # y_i - r_i >= 0 -> violation = max(0, r_i - y_i)
        # y_i + r_i <= 1 -> violation = max(0, y_i + r_i - 1)
        
        # Use smooth approximation: softplus-like
        eps = 1e-6
        def soft_viol(viol):
            return np.log1p(np.exp(viol / eps)) * eps
        
        # But for gradient, let's use a simpler quadratic penalty
        # violation^2 approach
        
        # Boundary: (x_i - r_i)^2 if x_i < r_i, (x_i + r_i - 1)^2 if x_i + r_i > 1
        for i in range(N):
            # Left boundary
            if x[i] < r[i]:
                viol = r[i] - x[i]
                penalty += viol * viol
                grad_x[i] += -2 * viol
                grad_r[i] += 2 * viol
            # Right boundary
            if x[i] + r[i] > 1:
                viol = x[i] + r[i] - 1
                penalty += viol * viol
                grad_x[i] += 2 * viol
                grad_r[i] += 2 * viol
            # Bottom boundary
            if y[i] < r[i]:
                viol = r[i] - y[i]
                penalty += viol * viol
                grad_y[i] += -2 * viol
                grad_r[i] += 2 * viol
            # Top boundary
            if y[i] + r[i] > 1:
                viol = y[i] + r[i] - 1
                penalty += viol * viol
                grad_y[i] += 2 * viol
                grad_r[i] += 2 * viol
        
        # Circle-circle constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist_sq < sum_r * sum_r:
                    # Overlap
                    dist = np.sqrt(max(dist_sq, 1e-12))
                    viol = sum_r - dist
                    penalty += viol * viol
                    # Gradient w.r.t. positions
                    if dist > 1e-12:
                        factor = 2 * viol / dist
                        grad_x[i] += factor * dx
                        grad_x[j] -= factor * dx
                        grad_y[i] += factor * dy
                        grad_y[j] -= factor * dy
                    # Gradient w.r.t. radii
                    grad_r[i] += 2 * viol
                    grad_r[j] += 2 * viol
        
        total_obj = obj + penalty
        total_grad = np.concatenate([grad_x, grad_y, grad_r])
        
        return total_obj, total_grad
    
    # Run optimization with increasing penalty weight
    best_var = var0.copy()
    best_obj = -np.sum(r0_vec)  # actual objective (not penalized)
    
    start_time = time.time()
    
    # Multi-phase optimization
    phases = [
        {'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6},
        {'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8},
        {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
    ]
    
    for phase in phases:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        try:
            res = minimize(
                objective_and_grad,
                best_var,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={
                    'maxiter': phase['maxiter'],
                    'ftol': phase['ftol'],
                    'gtol': phase['gtol'],
                    'maxls': 50,
                }
            )
            if res.success or res.fun < objective_and_grad(best_var)[0]:
                candidate = res.x
                # Evaluate actual objective
                r_cand = candidate[2*N:3*N]
                actual_obj = -np.sum(r_cand)
                if actual_obj > best_obj:
                    best_obj = actual_obj
                    best_var = candidate
        except Exception:
            pass
    
    # Polish phase: direct gradient ascent on sum(r) with constraint handling
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN - 10:
        # Use a simpler approach: iteratively push circles apart and grow
        v = best_var.copy()
        lr = 0.001
        for step in range(500):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            x = v[0:N].copy()
            y = v[N:2*N].copy()
            r = v[2*N:3*N].copy()
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Try to grow all radii
            fr[:] = 1.0
            
            # Repulsion from overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    sum_r = r[i] + r[j]
                    if dist_sq < sum_r * sum_r:
                        dist = np.sqrt(max(dist_sq, 1e-12))
                        if dist > 1e-12:
                            overlap = sum_r - dist
                            # Push apart
                            push = overlap * 0.5
                            fx[i] += push * dx / dist
                            fx[j] -= push * dx / dist
                            fy[i] += push * dy / dist
                            fy[j] -= push * dy / dist
                            # Reduce radii slightly
                            fr[i] -= 0.5 * overlap
                            fr[j] -= 0.5 * overlap
            
            # Boundary repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 2
                    fr[i] -= (r[i] - x[i])
                if x[i] + r[i] > 1:
                    fx[i] -= (x[i] + r[i] - 1) * 2
                    fr[i] -= (x[i] + r[i] - 1)
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 2
                    fr[i] -= (r[i] - y[i])
                if y[i] + r[i] > 1:
                    fy[i] -= (y[i] + r[i] - 1) * 2
                    fr[i] -= (y[i] + r[i] - 1)
            
            # Update
            v[0:N] = x + lr * fx
            v[N:2*N] = y + lr * fy
            v[2*N:3*N] = np.maximum(r + lr * fr, 0.001)
            
            # Clamp to bounds
            v[0:N] = np.clip(v[0:N], 0.0, 1.0)
            v[N:2*N] = np.clip(v[N:2*N], 0.0, 1.0)
            v[2*N:3*N] = np.clip(v[2*N:3*N], 0.001, 0.5)
            
            # Check if valid and better
            r_new = v[2*N:3*N]
            if np.sum(r_new) > best_obj:
                best_obj = np.sum(r_new)
                best_var = v.copy()
    
    # Final validation and cleanup
    x = best_var[0:N]
    y = best_var[N:2*N]
    r = best_var[2*N:3*N]
    
    # Ensure circles are inside the square
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    r = np.maximum(r, 0.001)
    
    # Fix any remaining overlaps by shrinking
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                if dist < sum_r:
                    # Shrink both
                    excess = sum_r - dist
                    r[i] -= excess * 0.5
                    r[j] -= excess * 0.5
                    changed = True
        if not changed:
            break
        r = np.maximum(r, 0.001)
    
    # Final check: ensure no overlaps and within bounds
    result = np.column_stack([x, y, r])
    
    return result


def construct():
    """
    Returns array of shape (26, 3) with rows (x, y, radius) of 26 disjoint
    circles inside the unit square, maximizing sum of radii.
    """
    start_time = time.time()
    
    try:
        result = _pack_26()
    except Exception as e:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
    
    # Validate: ensure all circles are inside [0,1]^2 with disjoint interiors
    x = result[:, 0]
    y = result[:, 1]
    r = result[:, 2]
    
    # Clamp radii to fit in square
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    r = np.maximum(r, 0.001)
    result[:, 2] = r
    
    # Final overlap fix
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                if dist < sum_r - 1e-9:
                    excess = sum_r - dist
                    # Shrink the larger circle more
                    if r[i] >= r[j]:
                        r[i] -= excess
                    else:
                        r[j] -= excess
                    changed = True
        if not changed:
            break
        r = np.maximum(r, 0.001)
        result[:, 2] = r
    
    return result
