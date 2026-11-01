"""Circle packing in perimeter-4 rectangle: SLSQP with Penalty Method + Polish + Epigraph."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _penalty_objective(z, penalty_weight=100.0):
    """
    z: [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1, W]
    H is derived as 2.0 - W.
    Objective: -sum(r) + penalty terms for constraint violations.
    """
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    W = z[3*N]
    H = 2.0 - W
    
    if W < 0.1 or H < 0.1:
        return 1e6

    obj = -np.sum(r)
    
    # Wall violations
    # x_i - r_i >= 0  -> violation = max(0, r_i - x_i)
    # W - x_i - r_i >= 0 -> violation = max(0, x_i + r_i - W)
    # y_i - r_i >= 0
    # H - y_i - r_i >= 0
    
    viol_x_left = np.maximum(r - x, 0.0)
    viol_x_right = np.maximum(x + r - W, 0.0)
    viol_y_bot = np.maximum(r - y, 0.0)
    viol_y_top = np.maximum(y + r - H, 0.0)
    
    wall_penalty = np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + \
                   np.sum(viol_y_bot**2) + np.sum(viol_y_top**2)
    
    # Pairwise separation violations
    # dist(i,j) >= r_i + r_j
    # violation = max(0, r_i + r_j - dist(i,j))
    
    # Vectorized pairwise calculation
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx*dx + dy*dy
    d = np.sqrt(np.maximum(d_sq, 0.0))
    
    r_sum = r[:, None] + r[None, :]
    
    # Only consider i < j to avoid double counting and self-interaction
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    r_sum_pairs = r_sum[iu]
    
    viol_pairs = np.maximum(r_sum_pairs - d_pairs, 0.0)
    pair_penalty = np.sum(viol_pairs**2)
    
    obj += penalty_weight * (wall_penalty + pair_penalty)
    
    return obj

def _polish_radii(x, y, W, H, r_init):
    """
    Given fixed positions and box, maximize radii.
    Iterative projection.
    """
    r = r_init.copy()
    # Precompute distances
    d_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_mat[i,j] = np.sqrt(dx*dx + dy*dy)
                
    # Wall limits
    wall_lim = np.minimum(np.minimum(x, W-x), np.minimum(y, H-y))
    
    for _ in range(200):
        improved = False
        for i in range(N):
            max_r = wall_lim[i]
            for j in range(N):
                if i != j:
                    allowed = d_mat[i,j] - r[j]
                    if allowed < max_r:
                        max_r = allowed
            if max_r < 0:
                max_r = 1e-9
            if max_r > r[i] + 1e-10:
                r[i] = max_r
                improved = True
        if not improved:
            break
    return r

def _polish_positions(x, y, r, W, H, max_iter=1000):
    """
    Given fixed radii and box, adjust positions to resolve overlaps and push to walls.
    Simple force-directed relaxation.
    """
    x = x.copy()
    y = y.copy()
    
    for _ in range(max_iter):
        moved = False
        for i in range(N):
            fx = 0.0
            fy = 0.0
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-8:
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    d = np.sqrt(dx*dx + dy*dy)
                r_sum = r[i] + r[j]
                if d < r_sum:
                    # Overlap, push apart
                    overlap = r_sum - d
                    fx += (dx / d) * overlap
                    fy += (dy / d) * overlap
            
            # Wall forces (push inward if outside, or just clamp later)
            # We'll just clamp after applying forces
            
            x[i] += fx
            y[i] += fy
            
            # Clamp to box
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])
            
            if np.abs(fx) > 1e-6 or np.abs(fy) > 1e-6:
                moved = True
        
        if not moved:
            break
            
    return x, y

def _epigraph_refine(x, y, r, W, H, time_limit=5.0):
    """
    Try to improve the packing by treating radii as variables to be maximized
    while ensuring feasibility. This is essentially re-running the optimization
    but starting from a very good point with tighter tolerances or different method.
    For now, we use a local SLSQP run with high precision on the penalty objective.
    """
    z = np.concatenate([x, y, r, [W]])
    
    start_t = time.time()
    deadline = start_t + time_limit
    
    # Use SLSQP with the penalty objective
    # We want to maximize sum(r) subject to constraints.
    # The penalty method handles constraints softly.
    # We'll run SLSQP to minimize -sum(r) + penalty.
    
    # To make it more effective, we can increase the penalty weight slightly
    # or just run SLSQP with tight ftol.
    
    try:
        res = minimize(
            lambda v: _penalty_objective(v, penalty_weight=50.0),
            z,
            method='SLSQP',
            bounds=[(1e-6, None)] * (3*N) + [(0.3, 1.7)],
            options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 50}
        )
        if res.success:
            x_new = res.x[0:N]
            y_new = res.x[N:2*N]
            r_new = res.x[2*N:3*N]
            W_new = res.x[3*N]
            H_new = 2.0 - W_new
            
            # Polish radii
            r_polished = _polish_radii(x_new, y_new, W_new, H_new, r_new)
            
            if _is_valid(x_new, y_new, r_polished, W_new, H_new):
                return x_new, y_new, r_polished, W_new, H_new
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, W, H):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, W, H):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), W, H)
                
    # Initializations
    aspect_deltas = [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.1, -0.1, 0.3, -0.3, 0.5, -0.5]
    
    attempts = 0
    max_attempts = 20
    
    for delta in aspect_deltas:
        if time.time() > deadline - 15:
            break
            
        W = 1.0 + delta
        H = 1.0 - delta
        
        if W < 0.3 or H < 0.3:
            continue
            
        # Grid placement
        cols = 5
        rows = 5
        xs = []
        ys = []
        for i in range(5):
            for j in range(5):
                cx = (i + 0.5) * (W / 5.0)
                cy = (j + 0.5) * (H / 5.0)
                xs.append(cx)
                ys.append(cy)
        
        x0 = np.array(xs[:21])
        y0 = np.array(ys[:21])
        
        # Add noise
        rng = np.random.default_rng(attempts)
        x0 = x0 + rng.uniform(-0.02, 0.02, N)
        y0 = y0 + rng.uniform(-0.02, 0.02, N)
        
        r_est = np.sqrt((W * H) / (N * 2 * np.sqrt(3)))
        r0 = np.full(N, r_est * 0.8)
        
        # Clip
        x0 = np.clip(x0, r0*0.5, W - r0*0.5)
        y0 = np.clip(y0, r0*0.5, H - r0*0.5)
        
        z0 = np.concatenate([x0, y0, r0, [W]])
        
        # Run SLSQP with penalty objective
        try:
            res = minimize(
                lambda v: _penalty_objective(v, penalty_weight=100.0),
                z0,
                method='SLSQP',
                bounds=[(1e-6, None)] * (3*N) + [(0.3, 1.7)],
                options={'maxiter': 300, 'ftol': 1e-10, 'maxls': 50}
            )
            
            x_res = res.x[0:N]
            y_res = res.x[N:2*N]
            r_res = res.x[2*N:3*N]
            W_res = res.x[3*N]
            H_res = 2.0 - W_res
            
            # Polish radii
            r_polished = _polish_radii(x_res, y_res, W_res, H_res, r_res)
            
            # Check validity and update
            if _is_valid(x_res, y_res, r_polished, W_res, H_res):
                update_best(x_res, y_res, r_polished, W_res, H_res)
            else:
                # Try to repair by shrinking radii
                for scale in [0.99, 0.98, 0.95]:
                    r_shrunk = r_polished * scale
                    if _is_valid(x_res, y_res, r_shrunk, W_res, H_res):
                        update_best(x_res, y_res, r_shrunk, W_res, H_res)
                        break
                        
        except Exception:
            pass
            
        attempts += 1
        
        # If we have time, try a few random restarts from the best solution found so far
        if best_result is not None and attempts < max_attempts and time.time() < deadline - 25:
            x_best, y_best, r_best, W_best, H_best = best_result
            for restart in range(3):
                if time.time() > deadline - 10:
                    break
                # Perturb best solution
                x_pert = x_best + np.random.uniform(-0.05, 0.05, N)
                y_pert = y_best + np.random.uniform(-0.05, 0.05, N)
                r_pert = r_best * 0.9 # Shrink to allow movement
                W_pert = W_best
                H_pert = H_best
                
                x_pert = np.clip(x_pert, r_pert, W_pert - r_pert)
                y_pert = np.clip(y_pert, r_pert, H_pert - r_pert)
                
                z_pert = np.concatenate([x_pert, y_pert, r_pert, [W_pert]])
                
                try:
                    res = minimize(
                        lambda v: _penalty_objective(v, penalty_weight=100.0),
                        z_pert,
                        method='SLSQP',
                        bounds=[(1e-6, None)] * (3*N) + [(0.3, 1.7)],
                        options={'maxiter': 200, 'ftol': 1e-10, 'maxls': 50}
                    )
                    x_res = res.x[0:N]
                    y_res = res.x[N:2*N]
                    r_res = res.x[2*N:3*N]
                    W_res = res.x[3*N]
                    H_res = 2.0 - W_res
                    
                    r_polished = _polish_radii(x_res, y_res, W_res, H_res, r_res)
                    
                    if _is_valid(x_res, y_res, r_polished, W_res, H_res):
                        update_best(x_res, y_res, r_polished, W_res, H_res)
                except Exception:
                    pass
                attempts += 1

    # Fallback if no valid result found
    if best_result is None:
        W, H = 1.0, 1.0
        cols = 5
        rows = 5
        xs = []
        ys = []
        for i in range(5):
            for j in range(5):
                cx = (i + 0.5) * (W / 5.0)
                cy = (j + 0.5) * (H / 5.0)
                xs.append(cx)
                ys.append(cy)
        x = np.array(xs[:21])
        y = np.array(ys[:21])
        r = np.full(N, 0.1)
        best_result = (x, y, r, W, H)
        
    x, y, r, W, H = best_result
    
    # Final polish: Epigraph refinement if time permits
    if time.time() < deadline - 10:
        refined = _epigraph_refine(x, y, r, W, H, time_limit=8.0)
        if refined is not None:
            x, y, r, W, H = refined
            
    # Final safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, W, H):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
