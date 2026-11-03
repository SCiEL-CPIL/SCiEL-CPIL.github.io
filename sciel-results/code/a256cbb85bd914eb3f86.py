"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _initial_grid(n):
    """Generate a grid initial configuration."""
    pos = []
    # Use 5x6 grid, take first n
    for i in range(5):
        for j in range(6):
            if len(pos) >= n:
                break
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 6.0
            pos.append([x, y])
    pos = np.array(pos[:n])
    radii = np.full(n, 0.05)
    return pos, radii

def _initial_hexagonal(n):
    """Generate a hexagonal initial configuration."""
    pos = []
    row_height = 0.18
    col_width = 0.18
    for r in range(7):
        for c in range(6):
            if len(pos) >= n:
                break
            x = (c + 0.5) * col_width
            y = (r + 0.5) * row_height
            if r % 2 == 1:
                x += col_width / 2
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pos.append([x, y])
    pos = np.array(pos[:n])
    radii = np.full(n, 0.05)
    return pos, radii

def _initial_random(n, seed=0):
    """Generate a random initial configuration with minimum separation."""
    rng = np.random.RandomState(seed)
    valid_pos = []
    attempts = 0
    while len(valid_pos) < n and attempts < 1000:
        p = rng.uniform(0.05, 0.95, size=2)
        ok = True
        for vp in valid_pos:
            if np.linalg.norm(p - vp) < 0.12:
                ok = False
                break
        if ok:
            valid_pos.append(p)
        attempts += 1
    while len(valid_pos) < n:
        valid_pos.append(rng.uniform(0.05, 0.95, size=2))
    pos = np.array(valid_pos)
    radii = np.full(n, 0.05)
    return pos, radii

def _penalty_and_objective(x, n, penalty_weight=100.0):
    """
    Compute objective (negative sum of radii) and penalty for violations.
    x = [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_n, y_n, r_n]
    """
    pos = x[0::3].reshape(-1, 2)
    radii = x[1::3]
    
    # Objective: maximize sum of radii
    obj = -np.sum(radii)
    
    # Penalty for boundary violations
    penalty = 0.0
    for i in range(n):
        x_i, y_i = pos[i]
        r_i = radii[i]
        # Boundary: circle must be inside [0,1]^2
        if x_i - r_i < 0:
            penalty += penalty_weight * (x_i - r_i) ** 2
        if y_i - r_i < 0:
            penalty += penalty_weight * (y_i - r_i) ** 2
        if x_i + r_i > 1:
            penalty += penalty_weight * (x_i + r_i - 1) ** 2
        if y_i + r_i > 1:
            penalty += penalty_weight * (y_i + r_i - 1) ** 2
    
    # Penalty for overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += penalty_weight * overlap ** 2
    
    return obj + penalty, obj

def _is_valid(pos, radii, n, tol=1e-9):
    """Check if packing is valid."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_to_boundary(pos, radii, n):
    """
    Fix invalid packing by reducing radii to satisfy constraints.
    Returns (pos, radii, is_valid)
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp to boundary
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
    
    # Fix overlaps iteratively
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                md = radii[i] + radii[j]
                if d < md - 1e-10:
                    excess = md - d
                    reduction = excess / 2.0 + 1e-8
                    radii[i] = max(0.0, radii[i] - reduction)
                    radii[j] = max(0.0, radii[j] - reduction)
                    changed = True
        if not changed:
            break
    
    # Final boundary clamp
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
    
    valid = _is_valid(pos, radii, n)
    return pos, radii, valid

def _optimize_with_scipy(pos, radii, n, deadline):
    """
    Optimize packing using scipy minimize with penalty method.
    """
    # Flatten: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pos[i, 0]
        x0[3*i+1] = pos[i, 1]
        x0[3*i+2] = radii[i]
    
    # Bounds: positions in [0,1], radii in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    best_result = None
    best_obj = np.inf
    
    # Try different penalty weights
    for pw in [50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline:
            break
        
        def objective(x):
            penalty_val, obj_val = _penalty_and_objective(x, n, penalty_weight=pw)
            return penalty_val
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_obj:
                x_opt = result.x
                pos_opt = x_opt[0::3].reshape(-1, 2)
                radii_opt = x_opt[1::3]
                
                # Fix to ensure validity
                pos_fixed, radii_fixed, valid = _fix_to_boundary(pos_opt, radii_opt, n)
                
                if valid:
                    current_obj = -np.sum(radii_fixed)
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_result = (pos_fixed, radii_fixed)
        except Exception:
            continue
    
    if best_result is not None:
        return best_result[0], best_result[1]
    else:
        return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_result = None
    best_score = -np.inf
    
    # Generate multiple initial configurations
    configs = []
    
    # Grid
    pos_grid, radii_grid = _initial_grid(N)
    configs.append((pos_grid, radii_grid))
    
    # Hexagonal
    pos_hex, radii_hex = _initial_hexagonal(N)
    configs.append((pos_hex, radii_hex))
    
    # Multiple random starts
    for seed in range(5):
        pos_rand, radii_rand = _initial_random(N, seed=seed)
        configs.append((pos_rand, radii_rand))
    
    # Process each configuration
    for pos, radii in configs:
        if time.time() > deadline - 15:
            break
        
        try:
            opt_pos, opt_radii = _optimize_with_scipy(pos, radii, N, deadline - 5)
            
            # Final validation and fix
            valid_pos, valid_radii, valid = _fix_to_boundary(opt_pos, opt_radii, N)
            
            if valid:
                score = np.sum(valid_radii)
                if score > best_score:
                    best_score = score
                    result = np.zeros((N, 3))
                    result[:, :2] = valid_pos
                    result[:, 2] = valid_radii
                    best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: small circles in grid
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(6):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 6.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        return result
    
    return best_result

# EVOLVE-BLOCK-END
