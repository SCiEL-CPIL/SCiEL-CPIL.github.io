import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _check_validity(circles, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _separate_circles(circles, W, H, max_iter=2000, tol=1e-12):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by clamping
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid

def _smooth_max(values, temperature=0.01):
    """Smooth max using log-sum-exp trick."""
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    if max_val > 50:
        return max_val
    shifted = values - max_val
    log_sum = max_val + np.log(np.sum(np.exp(shifted / temperature)) * temperature)
    return log_sum

def _objective_and_grad(params, W, H, beta=50.0):
    """
    Penalty-based objective with smooth max for constraints.
    params: flat array of shape (3*N,)
    Returns (objective_value, gradient)
    """
    n = N
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    grad = np.zeros(3 * n)
    grad[2::3] = -1.0
    
    # Penalty for boundary violations (smooth)
    boundary_violations = []
    for i in range(n):
        boundary_violations.append(r[i] - x[i])
        boundary_violations.append(r[i] - y[i])
        boundary_violations.append(r[i] - (W - x[i]))
        boundary_violations.append(r[i] - (H - y[i]))
    
    if boundary_violations:
        bv = np.array(boundary_violations)
        penalty_b = beta * _smooth_max(bv, 0.01)
        obj += penalty_b
        
        # Gradient of smooth max
        max_bv = np.max(bv)
        shifted_bv = bv - max_bv
        weights_bv = np.exp(shifted_bv / 0.01) / np.sum(np.exp(shifted_bv / 0.01))
        
        for idx, w in enumerate(weights_bv):
            if w < 1e-10:
                continue
            i = idx // 4
            which = idx % 4
            if which == 0:  # r - x < 0 violation
                grad[2*i] += beta * w
                grad[3*i] -= beta * w
            elif which == 1:  # r - y < 0
                grad[2*i+1] += beta * w
                grad[3*i+1] -= beta * w
            elif which == 2:  # r - (W-x) < 0
                grad[2*i] += beta * w
                grad[3*i] += beta * w
            elif which == 3:  # r - (H-y) < 0
                grad[2*i+1] += beta * w
                grad[3*i+1] += beta * w
    
    # Penalty for overlaps (smooth)
    overlap_violations = []
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            violation = min_dist - dist
            overlap_violations.append(violation)
            pair_indices.append((i, j, dx, dy, dist))
    
    if overlap_violations:
        ov = np.array(overlap_violations)
        penalty_o = beta * _smooth_max(ov, 0.01)
        obj += penalty_o
        
        max_ov = np.max(ov)
        shifted_ov = ov - max_ov
        weights_ov = np.exp(shifted_ov / 0.01) / np.sum(np.exp(shifted_ov / 0.01))
        
        for idx, w in enumerate(weights_ov):
            if w < 1e-10:
                continue
            i, j, dx, dy, dist = pair_indices[idx]
            if dist < 1e-12:
                continue
            ux = dx / dist
            uy = dy / dist
            # violation = (r_i + r_j) - dist
            # d(violation)/d(x_i) = -ux, d(violation)/d(x_j) = ux
            # d(violation)/d(r_i) = 1, d(violation)/d(r_j) = 1
            grad[3*i] -= beta * w * ux
            grad[3*i+1] -= beta * w * uy
            grad[3*j] += beta * w * ux
            grad[3*j+1] += beta * w * uy
            grad[3*i+2] += beta * w
            grad[3*j+2] += beta * w
    
    return obj, grad

def _local_optimize(circles, W, H, time_budget=10.0):
    """
    Gradient-based local optimization with adaptive penalty.
    """
    start = time.time()
    params = circles.flatten().copy()
    r_max = min(W, H) / 2
    
    best_params = params.copy()
    best_obj = _objective_and_grad(params, W, H, beta=50.0)[0]
    
    # Adaptive beta: start moderate, increase if violations persist
    betas = [20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() - start > time_budget - 1:
            break
            
        # Run L-BFGS-B with penalty objective
        try:
            result = minimize(
                lambda p: _objective_and_grad(p, W, H, beta=beta)[0],
                params,
                jac=lambda p: _objective_and_grad(p, W, H, beta=beta)[1],
                method='L-BFGS-B',
                bounds=[(1e-6, W-1e-6)] + [(1e-6, H-1e-6)] + [(1e-6, r_max)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            params = result.x
        except Exception:
            continue
        
        # Check if we improved
        obj_val = _objective_and_grad(params, W, H, beta=50.0)[0]
        if obj_val < best_obj:
            best_obj = obj_val
            best_params = params.copy()
    
    return best_params.reshape(N, 3)

def _polish(circles, W, H, max_iter=5000):
    """
    High-precision polish: alternate between gradient steps and separation.
    """
    params = circles.flatten().copy()
    
    for it in range(max_iter):
        # Gradient step with high beta for tight constraints
        obj, grad = _objective_and_grad(params, W, H, beta=1000.0)
        
        # Simple gradient descent with adaptive step size
        step_size = 1e-4
        for trial in range(10):
            new_params = params - step_size * grad
            new_obj, _ = _objective_and_grad(new_params, W, H, beta=1000.0)
            if new_obj < obj:
                params = new_params
                break
            step_size *= 0.5
        
        # Separation pass
        c = params.reshape(N, 3)
        c, valid = _separate_circles(c, W, H, max_iter=200)
        if valid:
            params = c.flatten()
    
    return params.reshape(N, 3)

def _hexagonal_initialization(W, H):
    """Create a hexagonal lattice initial configuration."""
    circles = []
    best_circles = None
    best_count = 0

    # Try a range of radii to fit 21 circles
    for trial_r in np.linspace(0.05, 0.30, 30):
        dx = 2 * trial_r * 1.001
        dy = 2 * trial_r * np.sqrt(3) / 2 * 1.001

        row = 0
        y_pos = trial_r + dy / 2
        while y_pos + trial_r <= H + 1e-9 and len(circles) < N:
            offset = 0.0 if row % 2 == 0 else dx / 2
            x_pos = trial_r + offset
            row += 1
            while x_pos + trial_r <= W + 1e-9 and len(circles) < N:
                circles.append([x_pos, y_pos, trial_r])
                x_pos += dx
            y_pos += dy

        if len(circles) > best_count:
            best_count = len(circles)
            best_circles = circles.copy()

        circles = []

    if best_circles is None or len(best_circles) == 0:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W, H) / (2 * side) - 1e-4
                circles.append([cx, cy, r])

    if len(circles) > N:
        circles = circles[:N]

    # If we have fewer than N, add random small ones or expand
    while len(circles) < N:
        # Find a spot for a small circle
        best_x, best_y, best_r = W/2, H/2, 0.01
        for _ in range(50):
            tx = np.random.uniform(0.05, W - 0.05)
            ty = np.random.uniform(0.05, H - 0.05)
            min_dist = min(W, H)
            for c in circles:
                d = np.sqrt((tx - c[0])**2 + (ty - c[1])**2)
                min_dist = min(min_dist, d - c[2])
            min_dist = min(min_dist, tx, W - tx, ty, H - ty)
            if min_dist > best_r:
                best_r = min_dist * 0.9
                best_x, best_y = tx, ty
        best_r = max(0.01, best_r)
        circles.append([best_x, best_y, best_r])

    return np.array(circles[:N])

def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    # 5x5 grid is 25, we need 21. 4x5 is 20, 5x5 is 25.
    # Let's try 5x5 and remove 4 corners or center? 
    # Or just 5x5 and let optimizer shrink/adjust.
    side_x = 5
    side_y = 5
    circles = []
    for i in range(side_x):
        for j in range(side_y):
            if len(circles) >= N:
                break
            cx = (i + 0.5) * W / side_x
            cy = (j + 0.5) * H / side_y
            r = min(W / side_x, H / side_y) / 2 - 1e-4
            circles.append([cx, cy, r])
    return np.array(circles[:N])

def _random_initialization(W, H, rng):
    """Create a random initial configuration."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)

def _solve_aspect(W, H, time_budget):
    """Solve for a specific aspect ratio."""
    start = time.time()
    best_circles = None
    best_score = -np.inf
    
    # Initialization strategies
    inits = []
    
    # 1. Hexagonal
    c_hex = _hexagonal_initialization(W, H)
    c_hex, valid_hex = _separate_circles(c_hex, W, H)
    if valid_hex:
        inits.append(c_hex)
        
    # 2. Grid
    c_grid = _grid_initialization(W, H)
    c_grid, valid_grid = _separate_circles(c_grid, W, H)
    if valid_grid:
        inits.append(c_grid)
        
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(3):
        c_rand = _random_initialization(W, H, rng)
        c_rand, valid_rand = _separate_circles(c_rand, W, H)
        if valid_rand:
            inits.append(c_rand)
            
    # Optimize each initialization
    for init in inits:
        if time.time() - start > time_budget - 2:
            break
            
        try:
            # Run gradient-based optimization
            t_remain = time_budget - (time.time() - start) - 1
            if t_remain < 1:
                break
            opt_circles = _local_optimize(init, W, H, time_budget=min(t_remain, 8.0))
            
            # Clean up and check validity
            opt_circles[:, 0] = np.clip(opt_circles[:, 0], opt_circles[:, 2], W - opt_circles[:, 2])
            opt_circles[:, 1] = np.clip(opt_circles[:, 1], opt_circles[:, 2], H - opt_circles[:, 2])
            opt_circles[:, 2] = np.clip(opt_circles[:, 2], 0.001, min(W, H) / 2)
            
            opt_circles, valid = _separate_circles(opt_circles, W, H)
            if valid:
                score = np.sum(opt_circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = opt_circles.copy()
        except Exception:
            continue
            
    # Polish the best result if time remains
    if best_circles is not None and time.time() - start < time_budget - 1:
        try:
            polished = _polish(best_circles, W, H, max_iter=2000)
            polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
            polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
            polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
            
            polished, valid = _separate_circles(polished, W, H, max_iter=1000)
            if valid:
                s = np.sum(polished[:, 2])
                if s > best_score:
                    best_score = s
                    best_circles = polished.copy()
        except Exception:
            pass
            
    return best_circles, best_score

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Aspect ratios to try
    # Perimeter = 2(W+H) = 4 => W+H=2 => H = 2-W
    # W ranges from 0.5 to 1.5
    # Square is W=1.0
    aspect_ratios = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    
    # Allocate time based on proximity to square, as square is often good
    # But we don't know which is best, so distribute evenly but prioritize center
    total_time = DEADLINE - 20 # Reserve 20s for final polish
    time_per_ar = total_time / len(aspect_ratios)
    
    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        W = 1.0 + (ar - 1.0) * 0.5 # Wait, ar = W/H?
        # Let's define ar = W/H.
        # W + H = 2. H = W/ar. W + W/ar = 2 => W(1+1/ar)=2 => W = 2ar/(ar+1)
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)
        
        # Time budget for this aspect ratio
        t_budget = min(time_per_ar, DEADLINE - 10 - elapsed)
        if t_budget < 1.0:
            continue
            
        circles, score = _solve_aspect(W, H, t_budget)
        
        if circles is not None and score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        try:
            t_remain = DEADLINE - 5 - (time.time() - start_time)
            if t_remain > 2:
                polished = _polish(best_result, best_W, best_H, max_iter=5000)
                polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
                polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
                polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
                
                polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=2000)
                if valid_final:
                    s = np.sum(polished[:, 2])
                    if s > best_score:
                        best_score = s
                        best_result = polished.copy()
        except Exception:
            pass

    # Final fallback: simple grid in square
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    # Ensure the result is valid; if not, fall back to a safe grid
    if not _check_validity(best_result, best_W, best_H):
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    return best_result
