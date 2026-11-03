import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 26

def _penalty_and_objective(params, N):
    """
    Computes the objective function to minimize.
    We want to maximize sum(r), so we minimize -sum(r).
    We add penalties for boundary violations and overlaps.
    params: array of shape (N, 3) with columns [x, y, r]
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Objective: Maximize sum of radii
    obj = -np.sum(r)
    
    # Penalty for boundary violations
    # x >= r, x <= 1-r, y >= r, y <= 1-r
    # Equivalent to: r <= min(x, 1-x, y, 1-y)
    # Violation = max(0, r - min(x, 1-x, y, 1-y))
    min_dist_to_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    boundary_violation = np.maximum(0, r - min_dist_to_boundary)
    penalty_boundary = np.sum(boundary_violation)
    
    # Penalty for overlaps
    # Distance d_ij must be >= r_i + r_j
    # Violation = max(0, (r_i + r_j) - d_ij)
    # Using squared penalty for smoothness: max(0, (r_i + r_j) - d_ij)^2
    penalty_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if d < min_dist:
                penalty_overlap += (min_dist - d)**2
                
    # Weights
    w_bound = 100.0
    w_overlap = 100.0
    
    return obj + w_bound * penalty_boundary + w_overlap * penalty_overlap

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Initial configuration: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    
    best_params = np.array(cells[:N])
    best_score = -np.sum(best_params[:, 2])  # We minimize -sum(r), so score is -obj
    
    # Helper to check validity and get score
    def get_score(params):
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Check boundaries
        if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5) or np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5):
            return -np.inf
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-5:
                    return -np.inf
        return np.sum(r)

    best_valid_score = get_score(best_params)
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout:
        # Create a perturbed initial guess
        params = best_params.copy()
        # Add small random noise to positions and radii
        noise_scale = 0.05
        params[:, 0] += rng.uniform(-noise_scale, noise_scale, N)
        params[:, 1] += rng.uniform(-noise_scale, noise_scale, N)
        params[:, 2] += rng.uniform(-noise_scale, noise_scale, N)
        
        # Clip to reasonable bounds to help optimizer
        params[:, 0] = np.clip(params[:, 0], 0.01, 0.99)
        params[:, 1] = np.clip(params[:, 1], 0.01, 0.99)
        params[:, 2] = np.clip(params[:, 2], 0.01, 0.4)
        
        # Optimize
        try:
            res = minimize(
                _penalty_and_objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if res.success:
                candidate = res.x.reshape(N, 3)
                # Ensure radii are positive
                candidate[:, 2] = np.maximum(candidate[:, 2], 1e-6)
                
                # Check validity
                score = get_score(candidate)
                if score > best_valid_score:
                    best_valid_score = score
                    best_params = candidate
        except Exception:
            pass
            
        # Check time
        if time.time() - start_time > timeout:
            break
            
    # Final cleanup: ensure strict validity by shrinking slightly if needed
    # The penalty method might leave tiny violations.
    # We can do a final "fix" pass:
    # 1. If any circle is out of bounds, shrink it or move it.
    # 2. If any circles overlap, shrink the smaller one or move them.
    
    # Simple iterative fix for final result
    final_params = best_params.copy()
    for _ in range(10):
        changed = False
        # Fix boundaries
        for i in range(N):
            min_dist = min(final_params[i, 0], 1 - final_params[i, 0], final_params[i, 1], 1 - final_params[i, 1])
            if final_params[i, 2] > min_dist:
                final_params[i, 2] = min_dist
                changed = True
        
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = final_params[i, 0] - final_params[j, 0]
                dy = final_params[i, 1] - final_params[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = final_params[i, 2] + final_params[j, 2]
                if d < min_dist:
                    # Shrink the smaller radius to resolve overlap
                    # Or scale both? Shrink smaller is safer for sum if one is much larger?
                    # Actually, to maximize sum, we should probably shrink both proportionally or just the smaller one.
                    # Let's shrink the smaller one to fit.
                    if final_params[i, 2] <= final_params[j, 2]:
                        # Reduce r_i so that r_i + r_j = d
                        new_r_i = d - final_params[j, 2]
                        if new_r_i > 0:
                            final_params[i, 2] = new_r_i
                        else:
                            # If d is too small, we must shrink both or move them.
                            # Moving is complex here. Let's just shrink both to half the distance.
                            final_params[i, 2] = d / 2 * 0.99
                            final_params[j, 2] = d / 2 * 0.99
                        changed = True
                    else:
                        new_r_j = d - final_params[i, 2]
                        if new_r_j > 0:
                            final_params[j, 2] = new_r_j
                        else:
                            final_params[i, 2] = d / 2 * 0.99
                            final_params[j, 2] = d / 2 * 0.99
                        changed = True
        if not changed:
            break
            
    return final_params
