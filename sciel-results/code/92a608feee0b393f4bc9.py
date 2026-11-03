"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _check_validity(arr):
    """Return True if all circles are inside [0,1]^2 with disjoint interiors."""
    if arr.shape != (N, 3):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # Check pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_sep = r[:, None] + r[None, :]
    # For i != j, dist must be >= min_sep (allow tiny tolerance)
    mask = ~np.eye(N, dtype=bool)
    if np.any(dist[mask] < min_sep[mask] - 1e-9):
        return False
    return True

def _objective(params):
    """Minimize negative sum of radii, with penalties for invalidity."""
    # params layout: for each circle i, [x_i, y_i, r_i]
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for going outside [0,1]^2
    pen = 0.0
    pen += np.sum(np.maximum(0, r - x)**2)
    pen += np.sum(np.maximum(0, r - (1 - x))**2)
    pen += np.sum(np.maximum(0, r - y)**2)
    pen += np.sum(np.maximum(0, r - (1 - y))**2)
    
    # Penalty for overlap
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_sep = r[:, None] + r[None, :]
    mask = ~np.eye(N, dtype=bool)
    overlap = np.maximum(0, min_sep - dist)
    pen += np.sum(overlap[mask]**2)
    
    # Return negative sum of radii plus penalty
    return -np.sum(r) + 10.0 * pen

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Initial grid-based seed
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    seed_arr = np.array(cells[:N])
    
    best_arr = seed_arr.copy()
    best_score = np.sum(seed_arr[:, 2])
    
    # Convert to flat parameter vector
    x0 = seed_arr.flatten()
    
    # Differential evolution for global search
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    # Actually radius can be up to 0.5, but let's be more permissive
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            init='sobol',
            polish=False,
            workers=1
        )
        
        if time.time() > deadline - 30:
            # Not enough time for refinement, return best from DE
            pass
        else:
            # Refine with L-BFGS-B
            result2 = minimize(
                _objective,
                result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result2.fun < result.fun:
                result = result2
        
        # Extract solution
        x = result.x[0::3]
        y = result.x[1::3]
        r = result.x[2::3]
        
        # Clip to valid range
        r = np.clip(r, 1e-6, 0.5)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        
        candidate = np.column_stack([x, y, r])
        
        # Verify validity
        if _check_validity(candidate):
            score = np.sum(candidate[:, 2])
            if score > best_score:
                best_score = score
                best_arr = candidate
        else:
            # Try to fix by reducing radii slightly
            r_fixed = r * 0.99
            x_fixed = np.clip(x, r_fixed, 1 - r_fixed)
            y_fixed = np.clip(y, r_fixed, 1 - r_fixed)
            candidate_fixed = np.column_stack([x_fixed, y_fixed, r_fixed])
            if _check_validity(candidate_fixed):
                score = np.sum(candidate_fixed[:, 2])
                if score > best_score:
                    best_score = score
                    best_arr = candidate_fixed
    
    except Exception:
        pass
    
    # If still at seed level, try a simple local improvement
    if best_score <= np.sum(seed_arr[:, 2]) + 1e-6:
        # Try to increase radii of the grid solution
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                # Try increasing radius of circle i
                for dr in [0.005, 0.002, 0.001]:
                    new_r = best_arr[i, 2] + dr
                    if new_r > 0.5:
                        continue
                    # Check if it fits
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = best_arr[i, 0] - best_arr[j, 0]
                        dy = best_arr[i, 1] - best_arr[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < new_r + best_arr[j, 2] - 1e-9:
                            valid = False
                            break
                    # Check bounds
                    if best_arr[i, 0] - new_r < -1e-9 or best_arr[i, 0] + new_r > 1 + 1e-9:
                        valid = False
                    if best_arr[i, 1] - new_r < -1e-9 or best_arr[i, 1] + new_r > 1 + 1e-9:
                        valid = False
                    
                    if valid:
                        best_arr[i, 2] = new_r
                        improved = True
                        break
            
            if not improved:
                break
    
    return best_arr

# EVOLVE-BLOCK-END
