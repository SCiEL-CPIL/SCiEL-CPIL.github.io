import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

N = 21

def _generate_hex_init(w, h, n_circles):
    """
    Generate initial circle positions in a hexagonal pattern.
    """
    circles = np.zeros((n_circles, 3))
    r_est = np.sqrt(w * h / (n_circles * np.pi)) * 0.95
    dx = 2.0 * r_est
    dy = np.sqrt(3) * r_est
    
    count = 0
    row = 0
    while count < n_circles:
        offset = (dx / 2) if (row % 2 == 1) else 0
        col = 0
        while count < n_circles:
            x = r_est + offset + col * dx
            y = r_est + row * dy
            if x + r_est <= w and y + r_est <= h:
                circles[count, 0] = x
                circles[count, 1] = y
                circles[count, 2] = r_est
                count += 1
            else:
                break
            col += 1
        row += 1
        if r_est + row * dy > h + r_est:
            break
    
    if count < n_circles:
        for i in range(count, n_circles):
            circles[i, 0] = w / 2
            circles[i, 1] = h / 2
            circles[i, 2] = r_est * 0.8
    
    return circles

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds.
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    if np.any(rs < 1e-9):
        return False
    
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
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
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
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
        
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
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
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return np.column_stack((pos, rad))

def _penalty_objective(x, w, h, penalty_weight):
    """
    Penalty-based objective for L-BFGS-B optimization.
    x: flattened [x0, y0, r0, x1, y1, r1, ...]
    Returns penalty value (lower is better).
    """
    n = len(x) // 3
    pos = x.reshape(n, 3)
    xs = pos[:, 0]
    ys = pos[:, 1]
    rs = pos[:, 2]
    
    # Sum of radii (we want to maximize, so negate)
    neg_sum_r = -np.sum(rs)
    
    # Boundary violations
    bound_viol = 0.0
    bound_viol += np.sum(np.maximum(0, rs - xs))  # x - r < 0
    bound_viol += np.sum(np.maximum(0, xs + rs - w))  # x + r > w
    bound_viol += np.sum(np.maximum(0, rs - ys))  # y - r < 0
    bound_viol += np.sum(np.maximum(0, ys + rs - h))  # y + r > h
    
    # Overlap violations (vectorized)
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist) ** 2
    
    # Negative radius penalty
    neg_r_pen = np.sum(np.maximum(0, -rs)) ** 2
    
    total_penalty = penalty_weight * (bound_viol + overlap_viol) + neg_r_pen
    return neg_sum_r + total_penalty

def _lbfgsb_optimize(circles, w, h, deadline, max_iter=5000):
    """
    Optimize using L-BFGS-B with penalty method.
    Much faster than SLSQP for this problem size.
    """
    if time.time() > deadline - 5.0:
        return None
    
    n = len(circles)
    x0 = circles.flatten()
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-8, w))  # x
        bounds.append((1e-8, h))  # y
        bounds.append((1e-8, min(w, h)/2))  # r
    
    best_x = x0.copy()
    best_obj = _penalty_objective(x0, w, h, 100.0)
    
    # Adaptive penalty weight
    penalty_weights = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    for pw in penalty_weights:
        if time.time() > deadline - 3.0:
            break
        
        def objective(x):
            if time.time() > deadline:
                return 0.0
            return _penalty_objective(x, w, h, pw)
        
        try:
            res = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter // len(penalty_weights), 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_obj:
                candidate = res.x
                # Validate with tight tolerance
                cand_circles = candidate.reshape(n, 3)
                if _validate_solution(cand_circles, w, h, N, tol=1e-6):
                    best_x = candidate
                    best_obj = res.fun
                else:
                    # Try to fix small violations
                    fixed = _resolve_collisions(cand_circles, w, h, max_iters=100)
                    if _validate_solution(fixed, w, h, N, tol=1e-6):
                        best_x = fixed.flatten()
                        best_obj = _penalty_objective(best_x, w, h, pw)
        except Exception:
            pass
    
    final_circles = best_x.reshape(n, 3)
    if _validate_solution(final_circles, w, h, N, tol=1e-6):
        return final_circles
    
    fixed = _resolve_collisions(final_circles, w, h, max_iters=500)
    if _validate_solution(fixed, w, h, N, tol=1e-6):
        return fixed
    
    return None

def _radial_chain_init(w, h, n_circles):
    """
    Generate initial positions using a radial chain parameterization.
    This creates a hexagonal-like packing with fewer degrees of freedom.
    """
    circles = np.zeros((n_circles, 3))
    
    # Start with hexagonal packing
    hex_init = _generate_hex_init(w, h, n_circles)
    
    # Refine with a few relaxation steps
    refined = _resolve_collisions(hex_init, w, h, max_iters=200)
    
    return refined

def _radial_optimize(circles, w, h, deadline):
    """
    Optimize using radial chain parameterization.
    Parameterize positions as perturbations from a hexagonal lattice.
    """
    if time.time() > deadline - 10.0:
        return None
    
    n = len(circles)
    
    # Use the current circles as starting point
    # Optimize with L-BFGS-B which is much faster
    result = _lbfgsb_optimize(circles, w, h, deadline, max_iter=8000)
    
    if result is not None:
        return result
    
    return None

def _slsqp_polish(circles, w, h, deadline):
    """
    Final high-precision polish with SLSQP.
    """
    if time.time() > deadline - 3.0:
        return circles
    
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
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-8})
    
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
    
    bounds = []
    for i in range(n):
        bounds.append((1e-8, w))
        bounds.append((1e-8, h))
        bounds.append((1e-8, min(w, h)/2))
    
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
                fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass
    
    return circles

def _generate_grid_init(w, h, n_circles, cols, rows):
    """
    Generate initial circle positions in a grid pattern.
    """
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
            
    return circles

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio using multiple strategies.
    """
    best_init = None
    best_init_score = 0.0
    
    # Try hexagonal initialization
    if time.time() < deadline - 15:
        hex_circles = _generate_hex_init(w, h, N)
        hex_relaxed = _resolve_collisions(hex_circles, w, h, max_iters=300)
        if _validate_solution(hex_relaxed, w, h, N):
            score = np.sum(hex_relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = hex_relaxed
    
    # Try grid initializations
    ar = w / h
    grid_configs = []
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6), (5, 4)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4), (4, 5)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4), (3, 7)]
        
    for cols, rows in grid_configs:
        if time.time() > deadline - 10:
            break
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=100)
        if _validate_solution(relaxed, w, h, N):
            score = np.sum(relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = relaxed
    
    if best_init is None:
        best_init = _generate_grid_init(w, h, N, 5, 5)
        best_init = _resolve_collisions(best_init, w, h, max_iters=100)
        if not _validate_solution(best_init, w, h, N):
            return None

    # Primary optimization: L-BFGS-B with penalty (fast)
    optimized = _radial_optimize(best_init, w, h, deadline)
    
    if optimized is None:
        optimized = best_init
    
    # Final polish with SLSQP (high precision, slower)
    if time.time() < deadline - 4:
        polished = _slsqp_polish(optimized, w, h, deadline)
        if _validate_solution(polished, w, h, N):
            if np.sum(polished[:, 2]) >= np.sum(optimized[:, 2]) - 1e-10:
                optimized = polished
        
        return optimized
    
    return optimized

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try - focus on promising ranges
    # Perimeter 4 => w + h = 2
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.95, 1.05]
    
    # Sort by expected promise (near-square first)
    aspect_ratios.sort(key=lambda x: abs(x - 1.0))
    
    for w in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        h = 2.0 - w
        
        time_remaining = deadline - time.time()
        if time_remaining < 15:
            continue
            
        try:
            result = _solve_aspect(w, h, deadline)
            if result is not None:
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass

    if best_result is None:
        # Fallback: Simple grid
        w, h = 1.0, 1.0
        positions = []
        radii = []
        cols = 5
        rows = 5
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.9
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])
        
    return best_result
