import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds.
    pos_r: (n, 3) array of [x, y, r]
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    # Check radii are positive
    if np.any(rs < 1e-9):
        return False
    
    # Check bounds: circles must be inside [0, w] x [0, h]
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    # Check non-overlap
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _resolve_collisions(circles, w, h, max_iters=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    circles: (n, 3) array [x, y, r]
    Returns modified circles array.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints: push circles inside [0, w] x [0, h]
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        
        # 2. Pairwise collisions: push apart
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        # Random perturbation if exactly overlapping
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart by half the overlap each
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    # Final clamp to ensure boundary validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return np.column_stack((pos, rad))

def _hexagonal_init(w, h, n_circles):
    """
    Generate initial circle positions in a hexagonal pattern.
    """
    circles = np.zeros((n_circles, 3))
    
    # Estimate side length for hex grid
    # Area per circle ~ (2*pi*r^2) / packing_density. 
    # Hex packing density ~ 0.9069.
    # Total Area = w*h.
    # n * 2*pi*r^2 / 0.9069 = w*h  => r = sqrt(w*h*0.9069 / (n * 2*pi))
    r_est = np.sqrt(w * h * 0.9069 / (n_circles * 2 * np.pi))
    
    # Hex grid spacing
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    count = 0
    row = 0
    while count < n_circles:
        for col in range(100):
            if count >= n_circles:
                break
            # Hex offset
            x_offset = 0 if row % 2 == 0 else r_est
            x = x_offset + col * dx + r_est
            y = row * dy + r_est
            
            # Check if inside bounds (roughly)
            if x + r_est > w or y + r_est > h:
                continue
                
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r_est
            count += 1
        row += 1
        
    # If we didn't fill enough, just fill the rest with small circles in remaining spots?
    # Or just resize the grid. For simplicity, if count < n, we might have issues.
    # But usually w*h is large enough.
    # If count < n_circles, we need to add more. 
    # Actually, the loop above might stop early if bounds are tight.
    # Let's just use a simple grid fallback if hex fails to fill.
    if count < n_circles:
         # Fallback to grid
         cols = int(np.ceil(np.sqrt(n_circles * w/h)))
         rows = int(np.ceil(n_circles / cols))
         for i in range(cols):
             for j in range(rows):
                 if count >= n_circles: break
                 x = (i + 0.5) * w / cols
                 y = (j + 0.5) * h / rows
                 r = 0.45 * min(w/cols, h/rows)
                 circles[count, 0] = x
                 circles[count, 1] = y
                 circles[count, 2] = r
                 count += 1
                 
    return circles[:n_circles]

def _penalty_optimize(circles, w, h, deadline):
    """
    Optimize using L-BFGS-B with penalty for constraints.
    """
    if time.time() > deadline - 10.0:
        return None

    n = len(circles)
    x0 = circles.flatten()
    
    # Penalty weight
    penalty_weight = 1000.0
    
    def objective_and_grad(x):
        if time.time() > deadline:
            return 0.0, np.zeros_like(x)
            
        pos = x[0::3]
        rad = x[1::3] # Wait, x is [x0, y0, r0, x1, y1, r1...]? 
        # No, x0 was circles.flatten() which is [x0, y0, r0, ...]
        # So pos is x[0::3], rad is x[2::3]
        
        # Correct indexing
        pos_x = x[0::3]
        pos_y = x[1::3]
        rad = x[2::3]
        
        # Objective: -sum(r)
        val = -np.sum(rad)
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        
        # Penalties
        # 1. Boundary
        # x >= r => x - r >= 0
        # x + r <= w => w - x - r >= 0
        # y >= r => y - r >= 0
        # y + r <= h => h - y - r >= 0
        
        # Soft plus: max(0, z) approx by 0.5*(z + sqrt(z^2 + eps))
        eps = 1e-4
        
        def soft_plus(z):
            return 0.5 * (z + np.sqrt(z*z + eps))
        
        # Compute violations
        v_x_min = rad - pos_x # violation if pos_x < rad
        v_x_max = pos_x + rad - w # violation if pos_x + rad > w
        v_y_min = rad - pos_y
        v_y_max = pos_y + rad - h
        
        pen_x_min = soft_plus(v_x_min)
        pen_x_max = soft_plus(v_x_max)
        pen_y_min = soft_plus(v_y_min)
        pen_y_max = soft_plus(v_y_max)
        
        total_pen = np.sum(pen_x_min) + np.sum(pen_x_max) + np.sum(pen_y_min) + np.sum(pen_y_max)
        val += penalty_weight * total_pen
        
        # Gradients for penalties
        # d/dz soft_plus(z) = 0.5 * (1 + z/sqrt(z^2+eps))
        
        def grad_soft_plus(z):
            return 0.5 * (1 + z / np.sqrt(z*z + eps))
        
        g_x_min = grad_soft_plus(v_x_min)
        g_x_max = grad_soft_plus(v_x_max)
        g_y_min = grad_soft_plus(v_y_min)
        g_y_max = grad_soft_plus(v_y_max)
        
        # d val / d pos_x
        # d pen_x_min / d pos_x = -g_x_min
        # d pen_x_max / d pos_x = g_x_max
        grad[0::3] += penalty_weight * (-g_x_min + g_x_max)
        
        # d val / d pos_y
        grad[1::3] += penalty_weight * (-g_y_min + g_y_max)
        
        # d val / d rad
        # d pen_x_min / d rad = g_x_min
        # d pen_x_max / d rad = g_x_max
        # d pen_y_min / d rad = g_y_min
        # d pen_y_max / d rad = g_y_max
        grad[2::3] += penalty_weight * (g_x_min + g_x_max + g_y_min + g_y_max)
        
        # 2. Overlaps
        # dist(i,j) >= r_i + r_j
        # violation = r_i + r_j - dist
        for i in range(n):
            for j in range(i+1, n):
                dx = pos_x[i] - pos_x[j]
                dy = pos_y[i] - pos_y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + eps) # Avoid div by 0
                min_dist = rad[i] + rad[j]
                
                v = min_dist - dist
                if v > 0:
                    # Soft plus
                    sv = soft_plus(v)
                    val += penalty_weight * sv
                    
                    gv = grad_soft_plus(v)
                    
                    # d sv / d rad_i = gv
                    # d sv / d rad_j = gv
                    # d sv / d pos_x_i = gv * (-dx/dist)
                    # d sv / d pos_x_j = gv * (dx/dist)
                    # ... and so on for y
                    
                    grad[2*i+2] += penalty_weight * gv
                    grad[2*j+2] += penalty_weight * gv
                    
                    dx_dist = dx / dist
                    dy_dist = dy / dist
                    
                    grad[2*i] += penalty_weight * (-gv * dx_dist)
                    grad[2*i+1] += penalty_weight * (-gv * dy_dist)
                    grad[2*j] += penalty_weight * (gv * dx_dist)
                    grad[2*j+1] += penalty_weight * (gv * dy_dist)

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r

    try:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_circles = res.x.reshape(-1, 3)
        
        # Check if valid
        if _validate_solution(new_circles, w, h, N):
            return new_circles
        else:
            # Try to fix
            fixed = _resolve_collisions(new_circles, w, h, max_iters=200)
            if _validate_solution(fixed, w, h, N):
                return fixed
    except Exception:
        pass
        
    return None

def _slsqp_polish(circles, w, h, deadline):
    """
    High precision polish with SLSQP.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)

    constraints = []

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )

        if res.success:
            new_circles = res.x.reshape(-1, 3)
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                fixed = _resolve_collisions(new_circles, w, h, max_iters=50)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass

    return None

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio.
    """
    # 1. Hex Init
    init_circles = _hexagonal_init(w, h, N)
    init_circles = _resolve_collisions(init_circles, w, h, max_iters=100)
    
    if not _validate_solution(init_circles, w, h, N):
        return None

    current_best = init_circles
    current_score = np.sum(init_circles[:, 2])

    # 2. Penalty Optimize
    if time.time() < deadline - 15.0:
        pen_res = _penalty_optimize(init_circles, w, h, deadline)
        if pen_res is not None:
            score = np.sum(pen_res[:, 2])
            if score > current_score:
                current_best = pen_res
                current_score = score

    # 3. SLSQP Polish
    if time.time() < deadline - 5.0:
        pol_res = _slsqp_polish(current_best, w, h, deadline)
        if pol_res is not None:
            score = np.sum(pol_res[:, 2])
            if score > current_score:
                current_best = pol_res
                current_score = score

    return current_best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin under 150s
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 1.3, 0.75, 1.5]
    
    # Sort by likelihood? Square is usually good start.
    # Let's just iterate.
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        result = _solve_aspect(w, h, deadline)
        
        if result is not None and _validate_solution(result, w, h, N):
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _hexagonal_init(w, h, N)
        best_result = _resolve_collisions(best_result, w, h, max_iters=100)
        if not _validate_solution(best_result, w, h, N):
            # Last resort: tiny circles
            best_result = np.zeros((N, 3))
            for i in range(N):
                x = (i % 5 + 0.5) * 0.2
                y = (i // 5 + 0.5) * 0.2
                best_result[i] = [x, y, 0.05]
            best_result = _resolve_collisions(best_result, w, h, max_iters=100)

    return best_result
