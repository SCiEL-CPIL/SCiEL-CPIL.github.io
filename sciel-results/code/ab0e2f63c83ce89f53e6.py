"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _objective(params, n_circles):
    """
    Objective: maximize sum of radii, with penalties for violations.
    params: flat array of [x0, y0, r0, x1, y1, r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for going outside [0,1]^2
    # x must be in [r, 1-r], y in [r, 1-r]
    penalty = 0.0
    for i in range(n_circles):
        # Lower bounds
        if x[i] < r[i]:
            penalty += (r[i] - x[i])**2
        if x[i] > 1 - r[i]:
            penalty += (x[i] - (1 - r[i]))**2
        if y[i] < r[i]:
            penalty += (r[i] - y[i])**2
        if y[i] > 1 - r[i]:
            penalty += (y[i] - (1 - r[i]))**2
        # Radius must be positive
        if r[i] < 0:
            penalty += r[i]**2
    
    # Penalty for overlapping circles
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    # We want to maximize sum(r), so minimize -sum(r) + penalty
    return -np.sum(r) + 100.0 * penalty

def _construct_grid(n):
    """Create a grid layout as initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return cells

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s to be safe before 120s limit
    
    n = N
    # Initial guess from grid
    grid_cells = _construct_grid(n)
    x0 = np.array([v for cell in grid_cells for v in cell])
    
    # Multiple restarts with different initializations
    best_params = x0.copy()
    best_score = _objective(x0, n)
    
    # Try several random perturbations of the grid
    rng = np.random.RandomState(42)
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            params = x0.copy()
        else:
            # Random perturbation
            params = x0.copy()
            # Perturb positions slightly
            params[0::3] += rng.uniform(-0.05, 0.05, n)
            params[1::3] += rng.uniform(-0.05, 0.05, n)
            # Perturb radii slightly
            params[2::3] *= rng.uniform(0.8, 1.2, n)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _objective,
                params,
                args=(n,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        if result.success or result.fun < best_score:
            # Check if the solution is valid (low penalty)
            score = result.fun
            x = result.x[0::3]
            y = result[1::3] if False else result.x[1::3]
            r = result.x[2::3]
            
            # Verify validity: check for overlaps and boundary violations
            valid = True
            for i in range(n):
                if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6:
                    valid = False
                    break
                if y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                    valid = False
                    break
                if r[i] < 1e-8:
                    valid = False
                    break
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Extract the actual sum of radii (negative of objective minus penalty)
                # Since penalty should be ~0 for valid solutions
                actual_score = -np.sum(r)
                if actual_score < best_score:  # We're minimizing, so lower is better
                    best_score = actual_score
                    best_params = result.x.copy()
        
        # Also check if result.fun is better regardless of strict validity
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    # Extract final result
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    # Clamp to ensure validity
    for i in range(n):
        r[i] = max(r[i], 1e-8)
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    # Final check: resolve any remaining overlaps by reducing radii
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j]:
                # Reduce radii to avoid overlap
                # Keep the larger one, reduce the smaller
                if r[i] >= r[j]:
                    r[j] = max(0.5 * dist - r[i], 1e-8)
                else:
                    r[i] = max(0.5 * dist - r[j], 1e-8)
                # Re-clamp
                x[i] = np.clip(x[i], r[i], 1 - r[i])
                y[i] = np.clip(y[i], r[i], 1 - r[i])
                x[j] = np.clip(x[j], r[j], 1 - r[j])
                y[j] = np.clip(y[j], r[j], 1 - r[j])
    
    result = np.zeros((n, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r
    
    return result
# EVOLVE-BLOCK-END
