# EVOLVE-BLOCK-START

import numpy as np
import time

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
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

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

def _gradient_descent(pos, radii, deadline, seed=0):
    """
    Gradient descent with projection to maximize sum of radii.
    This approach treats the problem as maximizing sum(r) subject to:
    1. dist(i,j) >= r_i + r_j
    2. x_i >= r_i, 1-x_i >= r_i, etc.
    
    We use a penalty method or direct projection. Here we use a direct 
    iterative adjustment:
    - Increase radii slightly.
    - If constraints are violated, move centers to separate circles or shrink radii.
    - Use gradient of the "violation" to guide center movement.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    
    # We want to maximize sum(radii).
    # Strategy: 
    # 1. Try to expand radii.
    # 2. If overlap occurs, push centers apart using gradient of overlap distance.
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    lr = 0.01
    expand_rate = 0.001
    
    for step in range(2000):
        if time.time() > deadline - 2:
            break
            
        # 1. Attempt to expand all radii uniformly
        radii += expand_rate
        radii = np.clip(radii, 1e-5, 0.5)
        
        # 2. Check constraints and calculate corrections
        # Boundary constraints
        # If x < r, push x right. If x > 1-r, push x left.
        # Overlap constraints
        # If dist < r_i + r_j, push apart.
        
        # Calculate gradients for positions to resolve violations
        grad_pos = np.zeros_like(pos)
        
        # Boundary gradients
        # For each circle i:
        # if pos[i,0] < radii[i]: grad_pos[i,0] += 1 (push right)
        # if pos[i,0] > 1 - radii[i]: grad_pos[i,0] -= 1 (push left)
        # Same for y
        
        # We can do this in a loop for clarity, or vectorize.
        # Given N=26, loop is fine.
        
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Boundaries
            if x < r:
                grad_pos[i, 0] += (r - x)
            if x > 1 - r:
                grad_pos[i, 0] -= (x - (1 - r))
            if y < r:
                grad_pos[i, 1] += (r - y)
            if y > 1 - r:
                grad_pos[i, 1] -= (y - (1 - r))
                
        # Overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r:
                    # Violation amount
                    viol = sum_r - dist
                    if dist < 1e-8:
                        # Random push
                        angle = rng.uniform(0, 2 * np.pi)
                        dx_dir = np.cos(angle)
                        dy_dir = np.sin(angle)
                    else:
                        dx_dir = dx / dist
                        dy_dir = dy / dist
                    
                    # Push i away from j, j away from i
                    grad_pos[i, 0] += viol * dx_dir
                    grad_pos[i, 1] += viol * dy_dir
                    grad_pos[j, 0] -= viol * dx_dir
                    grad_pos[j, 1] -= viol * dy_dir
        
        # Update positions
        pos += lr * grad_pos
        
        # Project positions back to valid range [0,1] roughly to keep them in bounds
        # Actually, we should project to the feasible region defined by radii?
        # No, just keep them in [0,1] and let the gradient handle the rest.
        # But if they go out, we need to clamp.
        pos = np.clip(pos, 0.0, 1.0)
        
        # If any circle is still significantly out of bounds or overlapping,
        # we might need to shrink radii to ensure validity for the next step.
        # However, the gradient step should resolve overlaps.
        
        # Check validity and update best
        # We need a strict check to ensure we don't return invalid.
        # But during optimization, we allow temporary violations?
        # The prompt says "disjoint interiors". 
        # If we return an invalid config, score is 0.
        # So we must ensure the returned config is valid.
        # The gradient step might leave small violations.
        # We will post-process at the end.
        
        # To ensure progress, we can also shrink radii if violations persist.
        # Let's check max violation.
        max_viol = 0
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            max_viol = max(max_viol, r - x, x - (1-r), r - y, y - (1-r))
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_viol = max(max_viol, radii[i] + radii[j] - dist)
        
        if max_viol > 0.01:
            # Shrink radii slightly to recover
            radii *= 0.99
            # And push positions back towards center of available space?
            # Or just let the next iteration handle it.
            
        # Update best if this configuration is valid (or close enough to be post-processable)
        # We'll just track the score and post-process at the end.
        # But we need to know if we are improving.
        # Let's calculate a "relaxed" score.
        
        # Periodic post-processing to ensure we don't drift into deep infeasibility
        if step % 50 == 0:
            p_proc, r_proc = _post_process(pos, radii)
            s = np.sum(r_proc)
            if s > best_score:
                best_score = s
                best_pos = p_proc
                best_radii = r_proc
                
    # Final post-process
    p_final, r_final = _post_process(pos, radii)
    s_final = np.sum(r_final)
    if s_final > best_score:
        best_score = s_final
        best_pos = p_final
        best_radii = r_final
        
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Grid Init + Gradient Descent
    if time.time() < deadline - 5:
        pos, radii = _init_grid()
        res = _gradient_descent(pos, radii, deadline, seed=0)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 2. Random Inits + Gradient Descent
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        pos, radii = _init_random(seed)
        res = _gradient_descent(pos, radii, deadline, seed=seed+100)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Local refinement on best solution (more aggressive GD)
    if best_pos is not None and time.time() < deadline - 5:
        # Run GD again on the best found, with different seed/noise
        refined_pos, refined_radii = _gradient_descent(best_pos, best_radii, deadline, seed=999)
        s = np.sum(refined_radii)
        if s > best_score:
            best_score = s
            best_pos = refined_pos
            best_radii = refined_radii

    if best_pos is None:
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
