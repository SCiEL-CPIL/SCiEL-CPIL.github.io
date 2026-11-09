import numpy as np
import time

N = 26

def _soft_overlap_penalty(x, y, r, beta=50.0):
    """
    Compute smooth overlap penalty using log-sum-exp (soft-max) formulation.
    Returns penalty value and gradients (grad_x, grad_y, grad_r).
    """
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    # violation = r_sum - d (positive means overlap)
    violation = r_sum - d
    
    # Soft-max: use log-sum-exp for smoothness
    # For each pair, we want max(0, violation) approximated smoothly
    # Use: soft_max(violation) = (1/beta) * log(sum(exp(beta * violation)))
    # But we only want the positive part, so we use a different approach
    
    # Actually, let's use a smooth approximation of max(0, v):
    # smooth_max(0, v) = (v + sqrt(v^2 + eps)) / 2
    eps = 1e-4
    smooth_viol = (violation + np.sqrt(violation**2 + eps)) / 2.0
    np.fill_diagonal(smooth_viol, 0.0)
    
    # Gradient of smooth_max(0, v) w.r.t. v is (1 + v/sqrt(v^2+eps)) / 2
    grad_v = (1.0 + violation / np.sqrt(violation**2 + eps)) / 2.0
    np.fill_diagonal(grad_v, 0.0)
    
    # d(smooth_viol) / d(x_i) = -grad_v_ij * (dx_ij / d_ij) for j != i
    # Sum over j
    d_safe = np.where(d < 1e-10, 1e-10, d)
    unit_x = dx / d_safe
    unit_y = dy / d_safe
    np.fill_diagonal(unit_x, 0.0)
    np.fill_diagonal(unit_y, 0.0)
    
    grad_x = -np.sum(grad_v * unit_x, axis=1)
    grad_y = -np.sum(grad_v * unit_y, axis=1)
    grad_r = np.sum(grad_v, axis=1)
    
    penalty = np.sum(smooth_viol)
    return penalty, grad_x, grad_y, grad_r

def _boundary_penalty(x, y, r):
    """
    Smooth boundary penalty.
    """
    # Distance to each boundary
    dist_left = x - r
    dist_right = 1.0 - x - r
    dist_bottom = y - r
    dist_top = 1.0 - y - r
    
    # Smooth max(0, -dist) = smooth violation
    eps = 1e-4
    def smooth_neg(v):
        # max(0, -v) = (-v + sqrt(v^2 + eps))/2
        return (-v + np.sqrt(v**2 + eps)) / 2.0
    
    def grad_smooth_neg(v):
        # d/dv [(-v + sqrt(v^2+eps))/2] = (-1 + v/sqrt(v^2+eps))/2
        return (-1.0 + v / np.sqrt(v**2 + eps)) / 2.0
    
    p_left = smooth_neg(dist_left)
    p_right = smooth_neg(dist_right)
    p_bottom = smooth_neg(dist_bottom)
    p_top = smooth_neg(dist_top)
    
    g_left = grad_smooth_neg(dist_left)
    g_right = grad_smooth_neg(dist_right)
    g_bottom = grad_smooth_neg(dist_bottom)
    g_top = grad_smooth_neg(dist_top)
    
    # d(p_left)/dx = g_left * d(dist_left)/dx = g_left * 1
    # d(p_right)/dx = g_right * d(dist_right)/dx = g_right * (-1)
    grad_x = g_left - g_right
    grad_y = g_bottom - g_top
    grad_r = -g_left - g_right - g_bottom - g_top
    
    penalty = np.sum(p_left + p_right + p_bottom + p_top)
    return penalty, grad_x, grad_y, grad_r

def _is_valid(x, y, r, tol=1e-9):
    """Check if configuration is valid."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    if np.any(d < r_sum - tol):
        return False
    
    if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
        return False
    
    if np.any(r < -tol):
        return False
    
    return True

def _enforce_validity(x, y, r):
    """Make configuration valid by shrinking radii as needed."""
    # First clamp to boundaries
    r_max_bound = np.minimum(
        np.minimum(x, 1 - x),
        np.minimum(y, 1 - y)
    )
    r = np.minimum(r, np.maximum(r_max_bound, 0.001))
    
    # Iteratively resolve overlaps
    for _ in range(500):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        overlaps = d < r_sum
        np.fill_diagonal(overlaps, False)
        
        if not np.any(overlaps):
            break
        
        # For each overlapping pair, shrink both circles
        for i in range(N):
            for j in range(i + 1, N):
                if overlaps[i, j]:
                    # Shrink both to just touch
                    excess = r[i] + r[j] - d[i, j]
                    if excess > 0:
                        # Split the reduction
                        r[i] -= excess / 2.0
                        r[j] -= excess / 2.0
                        r[i] = max(r[i], 0.001)
                        r[j] = max(r[j], 0.001)
        
        # Re-clamp to boundaries
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, np.maximum(r_max_bound, 0.001))
    
    return r

def _polish(x, y, r, time_budget=5.0):
    """
    High-precision polish: use local search to maximize sum of radii
    while maintaining validity.
    """
    start = time.time()
    
    # Try small perturbations and accept if valid and improves
    best_r = r.copy()
    best_score = np.sum(r)
    
    # Simulated annealing style local search
    np.random.seed(12345)
    step_size = 0.002
    iterations = 0
    
    while time.time() - start < time_budget and iterations < 20000:
        iterations += 1
        
        # Randomly pick a circle to move
        i = np.random.randint(N)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dr = np.random.uniform(-step_size, step_size)
        
        new_x = x[i] + dx
        new_y = y[i] + dy
        new_r = r[i] + dr
        
        # Check bounds
        if new_r < 0.005 or new_r > 0.4:
            continue
        if new_x - new_r < -1e-10 or new_x + new_r > 1 + 1e-10:
            continue
        if new_y - new_r < -1e-10 or new_y + new_r > 1 + 1e-10:
            continue
        
        # Check overlaps with all other circles
        valid = True
        for j in range(N):
            if j == i:
                continue
            dist = np.sqrt((new_x - x[j])**2 + (new_y - y[j])**2)
            if dist < new_r + r[j] - 1e-10:
                valid = False
                break
        
        if not valid:
            # Try shrinking slightly
            shrink = 0.95
            new_r2 = new_r * shrink
            if new_r2 < 0.005:
                continue
            if new_x - new_r2 < -1e-10 or new_x + new_r2 > 1 + 1e-10:
                continue
            if new_y - new_r2 < -1e-10 or new_y + new_r2 > 1 + 1e-10:
                continue
            valid = True
            for j in range(N):
                if j == i:
                    continue
                dist = np.sqrt((new_x - x[j])**2 + (new_y - y[j])**2)
                if dist < new_r2 + r[j] - 1e-10:
                    valid = False
                    break
        
        if valid:
            score = np.sum(r) - r[i] + new_r
            if score > best_score - 1e-12:
                x[i] = new_x
                y[i] = new_y
                r[i] = new_r
                best_score = score
                best_r = r.copy()
        
        # Decay step size
        step_size *= 0.9999
    
    return best_r

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_config = None
    
    # Multiple restarts with diverse initializations
    num_restarts = 40
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:  # Leave time for polish
            break
        
        np.random.seed(42 + restart * 13)
        
        if restart % 4 == 0:
            # Jittered grid - 5x5 grid with 26 circles
            side = 5
            xs = []
            ys = []
            rs = []
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < N:
                        x = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
                        y = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
                        x = np.clip(x, 0.1, 0.9)
                        y = np.clip(y, 0.1, 0.9)
                        xs.append(x)
                        ys.append(y)
                        rs.append(0.08)
                        count += 1
        elif restart % 4 == 1:
            # Random positions
            xs = np.random.uniform(0.15, 0.85, N)
            ys = np.random.uniform(0.15, 0.85, N)
            rs = np.full(N, 0.06)
        elif restart % 4 == 2:
            # Hexagonal-like packing pattern
            xs = []
            ys = []
            rs = []
            spacing = 0.18
            for row in range(5):
                for col in range(5):
                    if len(xs) < N:
                        x = 0.1 + col * spacing + (row % 2) * spacing * 0.5
                        y = 0.1 + row * spacing * 0.866
                        x += np.random.uniform(-0.02, 0.02)
                        y += np.random.uniform(-0.02, 0.02)
                        x = np.clip(x, 0.08, 0.92)
                        y = np.clip(y, 0.08, 0.92)
                        xs.append(x)
                        ys.append(y)
                        rs.append(0.07)
        else:
            # Clustered: some larger, some smaller
            xs = []
            ys = []
            rs = []
            for _ in range(6):
                x = np.random.uniform(0.25, 0.75)
                y = np.random.uniform(0.25, 0.75)
                r = np.random.uniform(0.09, 0.13)
                xs.append(x)
                ys.append(y)
                rs.append(r)
            for _ in range(N - 6):
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                r = np.random.uniform(0.03, 0.06)
                xs.append(x)
                ys.append(y)
                rs.append(r)
        
        x = np.array(xs, dtype=np.float64)
        y = np.array(ys, dtype=np.float64)
        r = np.array(rs, dtype=np.float64)
        
        # First, make it valid
        r = _enforce_validity(x, y, r)
        
        # Gradient-based optimization with smooth penalties
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 8.0  # Leave time for polish
        if remaining < 2.0:
            continue
        
        # Estimate steps: ~0.0015s per step
        steps = int(min(15000, max(3000, remaining / 0.0015)))
        
        lr = 0.003
        lambda_overlap = 10.0
        lambda_bound = 10.0
        
        for step in range(steps):
            if time.time() > deadline - 8.0:
                break
            
            # Compute smooth overlap penalty and gradients
            pen_ov, gx_ov, gy_ov, gr_ov = _soft_overlap_penalty(x, y, r)
            
            # Compute boundary penalty and gradients
            pen_bd, gx_bd, gy_bd, gr_bd = _boundary_penalty(x, y, r)
            
            # Objective: maximize sum(r) -> minimize -sum(r)
            gr_obj = -np.ones(N)
            
            # Total gradients
            grad_x = lambda_overlap * gx_ov + lambda_bound * gx_bd
            grad_y = lambda_overlap * gy_ov + lambda_bound * gy_bd
            grad_r = lambda_overlap * gr_ov + lambda_bound * gr_bd + gr_obj
            
            # Adaptive learning rate
            grad_mag = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2) + np.sum(grad_r**2))
            if grad_mag > 5.0:
                effective_lr = lr * (5.0 / grad_mag)
            else:
                effective_lr = lr
            
            # Update
            x -= effective_lr * grad_x
            y -= effective_lr * grad_y
            r -= effective_lr * grad_r
            
            # Clip
            r = np.maximum(r, 0.01)
            r = np.minimum(r, 0.4)
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            # Decay learning rate
            lr *= 0.9997
            
            # Adaptive penalty: reduce as we get valid
            if step % 1000 == 0 and step > 0:
                if pen_ov < 0.01 and pen_bd < 0.01:
                    lambda_overlap *= 0.8
                    lambda_bound *= 0.8
                elif pen_ov > 0.5:
                    lambda_overlap *= 1.2
                    lambda_bound *= 1.2
        
        # Enforce validity after optimization
        r = _enforce_validity(x, y, r)
        
        # Check validity
        if _is_valid(x, y, r, tol=1e-8):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = (x.copy(), y.copy(), r.copy())
        
        # If we have a good candidate, do polish
        if best_config is not None and time.time() < deadline - 5.0:
            bx, by, br = best_config
            polished_r = _polish(bx.copy(), by.copy(), br.copy(), time_budget=min(3.0, deadline - time.time() - 1.0))
            if _is_valid(bx, by, polished_r, tol=1e-8):
                polished_score = np.sum(polished_r)
                if polished_score > best_score:
                    best_score = polished_score
                    best_config = (bx.copy(), by.copy(), polished_r.copy())
    
    if best_config is None:
        # Fallback: simple grid
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.005
                    cells.append((x, y, r))
                    count += 1
        return np.array(cells)
    
    bx, by, br = best_config
    return np.column_stack((bx, by, br))
