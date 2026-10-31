# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii and adjust positions to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(500):
        valid = True
        
        # 1. Boundary constraints: shrink radius if it exceeds distance to wall
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                # Shrink to fit boundary with small margin
                new_r = max(max_r * 0.9999, 1e-6)
                if new_r < r:
                    radii[i] = new_r
                    valid = False
        
        # 2. Pairwise constraints: if overlapping, shrink both radii
        # We do this iteratively. For efficiency, we check all pairs.
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale down radii to fit distance
                    # Keep ratio of radii? Or just scale both?
                    # Scaling both by dist/sum_r makes them just touch.
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _penalty_objective(x, lambda_pen):
    """
    Objective function for Nelder-Mead.
    x: [x1, y1, ..., xN, yN, r1, ..., rN]
    Returns: -sum(radii) + lambda * sum(violations)
    """
    p = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    obj_val = -np.sum(r)
    
    # Violations
    # 1. Boundary violations
    # x - r >= 0 -> violation is max(0, r - x)
    # 1 - x - r >= 0 -> violation is max(0, x + r - 1)
    v_bound_x_neg = np.maximum(0, r - p[:, 0])
    v_bound_x_pos = np.maximum(0, p[:, 0] + r - 1.0)
    v_bound_y_neg = np.maximum(0, r - p[:, 1])
    v_bound_y_pos = np.maximum(0, p[:, 1] + r - 1.0)
    
    bound_violation = np.sum(v_bound_x_neg) + np.sum(v_bound_x_pos) + \
                      np.sum(v_bound_y_neg) + np.sum(v_bound_y_pos)
    
    # 2. Overlap violations
    # dist(p_i, p_j) >= r_i + r_j
    # violation = max(0, r_i + r_j - dist)
    dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
    dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Only consider upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_violation = np.sum(np.maximum(0, min_dist[mask] - dist[mask]))
    
    total_violation = bound_violation + overlap_violation
    
    return obj_val + lambda_pen * total_violation

def _local_search_nm(pos, radii, deadline, lambda_start=10.0):
    """
    Perform Nelder-Mead local search with adaptive penalty.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Initial penalty weight
    lam = lambda_start
    
    # We will run multiple iterations, reducing lambda if feasible, 
    # or just running a few NM steps.
    # Since NM is stochastic and can be slow, we limit iterations.
    
    best_x = x0.copy()
    best_f = _penalty_objective(x0, lam)
    
    # Bounds for NM are tricky, but we can clip inside the objective or 
    # rely on the penalty to keep it in range. 
    # However, NM can wander. We will clip x after each step? 
    # No, standard NM doesn't support bounds well. 
    # We will use a wrapper that clips x before evaluating? 
    # Actually, better to just let penalty handle it. 
    # But if x goes way out, dist calculations might be unstable? No.
    # Let's try standard NM.
    
    # To improve convergence, we can try different initial lambdas
    # or just run a long NM.
    
    # Use Powell or Nelder-Mead? 
    # Nelder-Mead is generally better for non-smooth (max) functions.
    
    while time.time() < deadline - 1:
        try:
            res = minimize(
                _penalty_objective, 
                best_x, 
                args=(lam,),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if res.fun < best_f:
                best_x = res.x
                best_f = res.fun
            
            # Check feasibility of the current best
            p = best_x[:2*N].reshape(N, 2)
            r = best_x[2*N:]
            # Quick validity check
            if _is_valid(p, r):
                # If valid, we can reduce lambda to prioritize radius maximization
                # or just stop if we've been stable?
                # Let's reduce lambda to explore larger radii
                lam = max(lam * 0.5, 0.1)
            else:
                # If invalid, increase lambda to enforce feasibility
                lam = min(lam * 2.0, 1000.0)
                
        except Exception:
            break
            
    # Final post-processing
    p = best_x[:2*N].reshape(N, 2)
    r = best_x[2*N:]
    p, r = _post_process(p, r)
    
    if _is_valid(p, r):
        return p, r
    return None

def _init_grid():
    """Grid-based initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.095
            k += 1
    if k < N:
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Grid Init + Local Search
    if time.time() < deadline - 10:
        pos, radii = _init_grid()
        res = _local_search_nm(pos, radii, deadline, lambda_start=10.0)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 2. Multiple Random Inits
    # We have ~100s left. Each NM run might take 5-10s? 
    # Let's try 10 random starts.
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        
        pos, radii = _init_random(seed)
        
        # Use a slightly higher lambda for random starts to ensure feasibility first
        res = _local_search_nm(pos, radii, deadline, lambda_start=20.0)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Refinement on the best solution found
    # Run a final NM with low lambda to squeeze out more radius
    if best_pos is not None and time.time() < deadline - 2:
        # Create a sub-deadline for this final refinement
        sub_deadline = time.time() + min(10.0, deadline - time.time() - 1.0)
        res = _local_search_nm(best_pos, best_radii, sub_deadline, lambda_start=1.0)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    if best_pos is None:
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
