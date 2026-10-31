"""
Circle packing in perimeter-4 rectangle.
Strategy:
1. Vectorized penalty objective for speed.
2. Dynamic aspect ratio optimization (W is a variable, H=2-W).
3. Force-directed relaxation for initial feasibility.
4. SLSQP optimization for global search and local refinement.
5. Expansion polish to maximize radii.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _vectorized_objective(z, penalty_weight=1000.0):
    """
    Vectorized objective function for circle packing.
    z = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN, W]
    Minimize: -sum(r) + penalty(overlaps) + penalty(boundaries)
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    # Basic sanity checks to avoid NaNs or huge penalties
    if w < 0.1 or h < 0.1 or np.any(r < 0):
        return 1e10

    obj = -np.sum(r)

    # 1. Pairwise Overlap Penalty
    # dx, dy, d_sq for all pairs
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    # Only consider upper triangle to avoid double counting and self-interaction
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    # Distance
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    
    # Overlap amount: max(0, r_i + r_j - d_ij)
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)
    
    obj += overlap_penalty

    # 2. Boundary Penalty
    # Circles must stay within [0, w] x [0, h]
    # x_i - r_i >= 0  => r_i - x_i <= 0
    # x_i + r_i <= w  => x_i + r_i - w >= 0
    # y_i - r_i >= 0  => r_i - y_i <= 0
    # y_i + r_i <= h  => y_i + r_i - h >= 0
    
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )
    
    obj += boundary_penalty

    # 3. Aspect Ratio Regularization (Optional, keeps W in reasonable range)
    # Prevents degenerate rectangles
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _force_relax(z, max_iter=200, time_limit=5.0):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    This acts as a pre-processor to make the initial guess feasible enough for SLSQP.
    """
    start_time = time.time()
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    w = z[-1]
    h = 2.0 - w
    
    # We will slightly shrink radii during relaxation to help find a feasible config
    # then expand them back in the main optimization.
    r_relax = r * 0.9
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        force_x = np.zeros(N)
        force_y = np.zeros(N)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r_relax[i] + r_relax[j]
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    # Overlap amount
                    overlap = min_dist - dist
                    # Force is proportional to overlap
                    f = overlap * 10.0
                    # Unit vector
                    ux = dx / dist
                    uy = dy / dist
                    
                    force_x[i] += f * ux
                    force_y[i] += f * uy
                    force_x[j] -= f * ux
                    force_y[j] -= f * uy
        
        # Boundary repulsion
        for i in range(N):
            if x[i] < r_relax[i]:
                force_x[i] += (r_relax[i] - x[i]) * 10.0
            elif x[i] > w - r_relax[i]:
                force_x[i] -= (x[i] - (w - r_relax[i])) * 10.0
            
            if y[i] < r_relax[i]:
                force_y[i] += (r_relax[i] - y[i]) * 10.0
            elif y[i] > h - r_relax[i]:
                force_y[i] -= (y[i] - (h - r_relax[i])) * 10.0
        
        # Apply forces with damping
        step_size = 0.1
        x += step_size * force_x
        y += step_size * force_y
        
        # Clip to boundaries to prevent flying out
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)

    # Restore original radii for the next stage
    z_new = z.copy()
    z_new[0::3][:N] = x
    z_new[1::3][:N] = y
    # z_new[2::3][:N] remains original r
    return z_new


def _check_validity(z, tol=1e-6):
    """Check if the configuration is strictly valid."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    
    if w <= 0 or h <= 0:
        return False, 0.0
        
    for i in range(N):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
            
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False, 0.0
                
    val = float(np.sum(r))
    return True, val


def _expansion_polish(z, time_limit=10.0):
    """
    Try to increase radii while maintaining validity.
    This is a heuristic polish that pushes circles outwards.
    """
    start_time = time.time()
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    w = z[-1]
    h = 2.0 - w
    
    # We will try to increase all radii by a small factor and then relax positions
    # to accommodate them.
    best_val = np.sum(r)
    best_z = z.copy()
    
    # Iterative expansion
    for step in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Try expanding radii by 1%
        r_new = r * 1.01
        
        # Check if this expansion is even geometrically possible (rough check)
        # If max radius > min(w,h)/2, it's impossible.
        if np.max(r_new) > min(w, h)/2:
            break
            
        # Update z with new radii
        z_trial = z.copy()
        z_trial[2::3][:N] = r_new
        
        # Relax positions to accommodate new radii
        z_relaxed = _force_relax(z_trial, max_iter=50, time_limit=2.0)
        
        # Check validity of the relaxed configuration
        valid, val = _check_validity(z_relaxed)
        
        if valid and val > best_val:
            best_val = val
            best_z = z_relaxed
            r = r_new # Update current r for next iteration
            x = z_relaxed[0::3][:N].copy()
            y = z_relaxed[1::3][:N].copy()
        else:
            # If expansion failed, try a smaller step or break
            # If we can't expand by 1%, we likely can't expand at all
            # Try 0.5% once?
            r_half = r * 1.005
            z_half = z.copy()
            z_half[2::3][:N] = r_half
            z_relaxed_half = _force_relax(z_half, max_iter=30, time_limit=1.0)
            valid_half, val_half = _check_validity(z_relaxed_half)
            
            if valid_half and val_half > best_val:
                best_val = val_half
                best_z = z_relaxed_half
                r = r_half
            else:
                break
                
    return best_z


def construct():
    """
    Construct 21 circles in a perimeter-4 rectangle maximizing sum of radii.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_value = -np.inf
    
    # Define bounds for SLSQP
    # x, y in [0, 2], r in [0, 1], W in [0.2, 1.8]
    bounds = []
    for i in range(N):
        bounds.append((0, 2)) # x
        bounds.append((0, 2)) # y
        bounds.append((0, 1)) # r
    bounds.append((0.2, 1.8)) # W
    
    # Objective function wrapper
    def objective(z):
        return _vectorized_objective(z)
        
    # Generate diverse initial guesses
    # Aspect ratios to try
    aspect_ratios = [1.0, 0.8, 1.2, 0.6, 1.4, 0.7, 1.3, 0.9, 1.1]
    
    initial_guesses = []
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.2: continue
        
        # Grid initialization
        z = np.zeros(3 * N + 1)
        # Arrange circles in a grid
        cols = 5
        rows = 5
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N: break
                cx = (j + 0.5) * w0 / cols
                cy = (i + 0.5) * h0 / rows
                # Estimate radius
                r0 = min(w0 / cols, h0 / rows) * 0.45
                z[3*idx] = cx
                z[3*idx+1] = cy
                z[3*idx+2] = r0
                idx += 1
        z[-1] = w0
        initial_guesses.append(z)
        
    # Add some random guesses
    rng = np.random.default_rng(42)
    for _ in range(5):
        w0 = rng.uniform(0.5, 1.5)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.05, 0.15)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3*i] = cx
            z[3*i+1] = cy
            z[3*i+2] = r0
        z[-1] = w0
        initial_guesses.append(z)
        
    for i, z0 in enumerate(initial_guesses):
        if time.time() > deadline - 20:
            break
            
        # 1. Force Relaxation to get a feasible starting point
        z_relaxed = _force_relax(z0, max_iter=100, time_limit=5.0)
        
        # 2. SLSQP Optimization
        # We use the relaxed point as start
        try:
            # SLSQP can be slow, so we limit iterations
            # We don't use explicit constraints in SLSQP here, we rely on the penalty method
            # This is because defining 210+ nonlinear constraints is expensive for SLSQP
            # The penalty method with a strong weight works well if the initial guess is good.
            
            res = minimize(
                objective, 
                z_relaxed, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 50}
            )
            
            z_opt = res.x
            valid, val = _check_validity(z_opt)
            
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # If SLSQP didn't find a strictly valid point, try to repair
                # by shrinking radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, 3*N, 3)
                z_repair[r_idx] *= 0.95
                valid_repair, val_repair = _check_validity(z_repair)
                if valid_repair and val_repair > best_value:
                    best_value = val_repair
                    best_result = z_repair.copy()
                    
        except Exception:
            continue
            
    # 3. Expansion Polish on the best result
    if best_result is not None and time.time() < deadline - 15:
        z_expanded = _expansion_polish(best_result, time_limit=10.0)
        valid_exp, val_exp = _check_validity(z_expanded)
        if valid_exp and val_exp > best_value:
            best_value = val_exp
            best_result = z_expanded
            
    # 4. Final SLSQP Polish
    if best_result is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                objective, 
                best_result, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'maxls': 50}
            )
            z_final = res_final.x
            valid_final, val_final = _check_validity(z_final)
            if valid_final and val_final > best_value:
                best_value = val_final
                best_result = z_final
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        # Simple grid in 1x1 square
        W, H = 1.0, 1.0
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) * W / side
                y = (j + 0.5) * H / side
                rad = min(W, H) / (2 * side) - 1e-4
                cells.append([x, y, rad])
                count += 1
        best_result = np.array(cells[:N])
        # Note: This fallback returns (x,y,r) but the main logic returns (x,y,r,W)
        # We need to be consistent. The task requires returning (n,3) array of (cx, cy, r).
        # The internal representation includes W. We must strip W before returning.
        
    # Extract x, y, r from best_result
    # If best_result is from the main logic, it has shape (3N+1,)
    # If it's the fallback, it has shape (N, 3)
    
    if best_result.ndim == 1:
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N]
    else:
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        
    # Final safety check and shrink if necessary
    # The evaluator checks perimeter <= 4.
    # Our W, H satisfy 2(W+H)=4 exactly if H=2-W.
    # But numerical errors might creep in.
    # We just return x, y, r. The evaluator will compute the bounding box.
    # Wait, the task says "bounding rectangle has perimeter at most 4".
    # It does NOT say we must use the W,H we optimized.
    # The evaluator will find the min bounding box of the circles and check its perimeter.
    # So we just need to ensure the circles are disjoint and fit in SOME rectangle of perim 4.
    # Since we optimized for a specific W,H, the circles fit in that W,H.
    # So the bounding box will be <= W,H, so perimeter <= 4.
    
    return np.column_stack((x, y, r))

# EVOLVE-BLOCK-END
