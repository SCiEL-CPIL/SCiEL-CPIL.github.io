import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _compute_radius(c):
    """
    Compute the true radius and check validity.
    c: (N, 2) array
    """
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Check bounds
    if np.any(np.abs(c) > 0.5 + 1e-9):
        return 0.0
        
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    min_pair_dist = np.min(dists[mask])
    
    # Boundary distances
    # min(0.5 - x, 0.5 + x) for each coordinate
    boundary_dists = np.min(0.5 - np.abs(c), axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    if r < 0:
        return 0.0
        
    return r

def _soft_min_obj_and_grad(c_flat, beta):
    """
    Objective: -S(c) where S(c) is the soft-min approximation of the min distance.
    c_flat: (2*N,) array
    """
    c = c_flat.reshape(N, 2)
    
    # 1. Compute distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero in gradient calculation if dist is 0
    eps = 1e-12
    dists_safe = np.maximum(dists, eps)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Pairwise part
    exponents_p = -beta * dists
    exponents_p[~mask] = -np.inf # Ignore diagonal
    
    max_exp_p = np.max(exponents_p)
    if np.isinf(max_exp_p):
        return 1e10, np.zeros_like(c_flat)
        
    sum_exp_p = np.sum(np.exp(exponents_p - max_exp_p))
    if sum_exp_p == 0:
        return 1e10, np.zeros_like(c_flat)
        
    U = -1.0/beta * (np.log(sum_exp_p) + max_exp_p)
    
    weights_p = np.exp(exponents_p - max_exp_p) / sum_exp_p
    
    # Boundary part
    b_ik = 0.5 - np.abs(c)
    
    exponents_b = -beta * b_ik
    max_exp_b = np.max(exponents_b)
    if np.isinf(max_exp_b):
        return 1e10, np.zeros_like(c_flat)
        
    sum_exp_b = np.sum(np.exp(exponents_b - max_exp_b))
    if sum_exp_b == 0:
        return 1e10, np.zeros_like(c_flat)
        
    V = -1.0/beta * (np.log(sum_exp_b) + max_exp_b)
    
    weights_b = np.exp(exponents_b - max_exp_b) / sum_exp_b
    
    # Combine U and V into S
    exp_U = np.exp(-beta * U)
    exp_V = np.exp(-beta * V)
    
    denom = exp_U + exp_V
    if denom == 0:
        return 1e10, np.zeros_like(c_flat)
        
    S = -1.0/(2.0*beta) * np.log(denom)
    
    # Gradient
    grad_U = np.zeros_like(c)
    grad_V = np.zeros_like(c)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_dists = diff / dists_safe[:, :, np.newaxis]
        
    grad_U = np.sum(weights_p[:, :, np.newaxis] * grad_dists, axis=1)
    
    sign_c = np.sign(c)
    grad_b = -sign_c
    grad_V = np.sum(weights_b[:, :, np.newaxis] * grad_b, axis=1)
    
    grad_S = 0.5 * (exp_U * grad_U + exp_V * grad_V) / denom
    
    obj = -S
    grad = -grad_S.flatten()
    
    return obj, grad

def _construct_grid():
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _construct_hex():
    # Hexagonal packing initialization
    # For N=50, a good hexagonal arrangement
    # Let's try to place them in rows
    # Row lengths: 7, 6, 7, 6, 7, 6, 7, 4 (sum=50)
    # Or 8, 7, 7, 7, 7, 7, 7 (sum=50)
    # Let's use a simple hex grid and take the first 50
    
    pts = []
    # Generate a hex grid
    spacing_x = 1.0
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    for row in range(10):
        for col in range(10):
            x = col * spacing_x
            y = row * spacing_y
            if row % 2 == 1:
                x += spacing_x / 2
            pts.append([x, y])
            
    pts = np.array(pts)
    
    # Normalize to fit in [-0.5, 0.5]
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    scale = 1.0 / max(width, height)
    pts = (pts - (min_x, min_y)) * scale - 0.5
    
    # Add small random perturbation
    pts = pts + np.random.normal(0, 0.01, (N, 2))
    return pts

def _construct_random():
    return np.random.uniform(-0.5, 0.5, (N, 2))

def _feasibility_check(c, r_target):
    """
    Check if circles of radius r_target can fit with centers c.
    Returns (feasible, min_violation)
    """
    # Check boundary
    boundary_violation = np.max(np.abs(c) + r_target - 0.5)
    
    # Check pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    min_pair_dist = np.min(dists[mask])
    pair_violation = 2 * r_target - min_pair_dist
    
    min_violation = max(boundary_violation, pair_violation)
    feasible = min_violation <= 1e-9
    
    return feasible, min_violation

def _push_apart(c, r_target, n_steps=100, step_size=0.01):
    """
    Push overlapping circles apart using projected gradient descent.
    This is a constraint satisfaction approach.
    """
    c = c.copy()
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        
        # For overlapping pairs, push apart
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_dists = diff / np.maximum(dists, 1e-12)[:, :, np.newaxis]
        
        # Only consider overlapping pairs
        overlap_mask = (dists < 2 * r_target) & mask
        forces += np.sum(np.where(overlap_mask[:, :, np.newaxis], grad_dists, 0), axis=1)
        
        # Boundary constraint
        # Push away from walls
        for i in range(N):
            for k in range(2):
                if c[i, k] + r_target > 0.5:
                    forces[i, k] -= 1.0
                if c[i, k] - r_target < -0.5:
                    forces[i, k] += 1.0
        
        # Update
        c += step_size * forces
        
        # Project back to square
        c = np.clip(c, -0.5, 0.5)
    
    return c

def _binary_search_radius(c_init, r_low, r_high, max_iters=20):
    """
    Binary search on target radius to find the maximum feasible radius.
    """
    best_r = r_low
    best_c = c_init.copy()
    
    for _ in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        r_mid = (r_low + r_high) / 2
        
        # Try to find a feasible configuration for r_mid
        # Start from current best_c
        c_trial = best_c.copy()
        
        # Push apart to satisfy constraints
        c_trial = _push_apart(c_trial, r_mid, n_steps=50, step_size=0.005)
        
        # Check feasibility
        feasible, violation = _feasibility_check(c_trial, r_mid)
        
        if feasible:
            best_r = r_mid
            best_c = c_trial
            r_low = r_mid
        else:
            r_high = r_mid
    
    return best_r, best_c

def construct():
    best_c = _construct_grid()
    best_r = _compute_radius(best_c)
    
    # Try different initializations
    initializers = [
        _construct_grid(),
        _construct_hex(),
        _construct_random(),
        _construct_random(),
        _construct_random()
    ]
    
    # First, use soft-min optimization to get a good starting point
    betas = [50, 100, 200, 500, 1000, 2000]
    
    for init_c in initializers:
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
            
        current_c = init_c.copy()
        
        for beta in betas:
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
                
            c_flat = current_c.flatten()
            
            try:
                bounds = [(-0.5, 0.5)] * (2*N)
                
                res = minimize(
                    _soft_min_obj_and_grad,
                    c_flat,
                    method='L-BFGS-B',
                    jac=True,
                    args=(beta,),
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                new_c = res.x.reshape(N, 2)
                new_r = _compute_radius(new_c)
                
                if new_r > best_r:
                    best_r = new_r
                    best_c = new_c.copy()
                    
                current_c = new_c
                
            except Exception:
                pass
    
    # Now use binary search with constraint satisfaction to refine
    if time.time() - START_TIME < TIME_LIMIT - 15:
        # Use current best as starting point
        r_low = best_r
        r_high = best_r * 1.1  # Try to improve by 10%
        
        # Binary search
        for _ in range(10):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            r_mid = (r_low + r_high) / 2
            
            # Try multiple restarts from perturbed best_c
            success = False
            for restart in range(3):
                c_trial = best_c + np.random.normal(0, 0.01, (N, 2))
                c_trial = np.clip(c_trial, -0.5, 0.5)
                
                c_trial = _push_apart(c_trial, r_mid, n_steps=100, step_size=0.005)
                
                feasible, violation = _feasibility_check(c_trial, r_mid)
                
                if feasible:
                    new_r = _compute_radius(c_trial)
                    if new_r > best_r:
                        best_r = new_r
                        best_c = c_trial.copy()
                    success = True
                    break
            
            if success:
                r_low = r_mid
            else:
                r_high = r_mid
    
    # Final polish with high beta
    if time.time() - START_TIME < TIME_LIMIT - 2:
        c_flat = best_c.flatten()
        bounds = [(-0.5, 0.5)] * (2*N)
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                c_flat,
                method='L-BFGS-B',
                jac=True,
                args=(5000,),
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_c = res.x.reshape(N, 2)
            new_r = _compute_radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_c = new_c
        except:
            pass
            
    return best_c

if __name__ == "__main__":
    c = construct()
    r = _compute_radius(c)
    print(f"Radius: {r}")

