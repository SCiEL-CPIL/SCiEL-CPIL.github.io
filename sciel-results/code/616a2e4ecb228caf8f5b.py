import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _force_directed_pack(circles, w, h, max_iters=500, tol=1e-9):
    """
    Force-directed relaxation to form natural contact structures.
    Uses spring forces for overlaps and wall forces.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    # Temperature schedule for simulated annealing-like behavior
    temp = 0.1
    cooling = 0.995
    
    for it in range(max_iters):
        if time.time() > deadline_global - 10.0:
            break
            
        forces = np.zeros_like(pos)
        
        # Wall forces: push circles away from walls
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x < r:
                forces[i, 0] += (r - x) * 10.0
            elif x > w - r:
                forces[i, 0] -= (x - (w - r)) * 10.0
            if y < r:
                forces[i, 1] += (r - y) * 10.0
            elif y > h - r:
                forces[i, 1] -= (y - (h - r)) * 10.0
        
        # Pairwise repulsion: spring force when overlapping
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Hooke's law: F = k * (overlap)
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    force_mag = overlap * 5.0
                    forces[i, 0] -= nx * force_mag
                    forces[i, 1] -= ny * force_mag
                    forces[j, 0] += nx * force_mag
                    forces[j, 1] += ny * force_mag
        
        # Apply forces with temperature-controlled step size
        step = temp * 0.1
        pos += forces * step
        
        # Clamp to bounds
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
            pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
        # Cool down
        temp *= cooling
        
        # Check convergence
        if np.max(np.abs(forces)) < tol:
            break
    
    return np.column_stack((pos, rad))

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
            # Initial radius: half the min cell size, slightly smaller
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
            
    return circles

def _slsqp_optimize(circles, w, h, deadline, maxiter=200):
    """
    Optimize radii and positions using SLSQP with smooth surrogate constraints.
    Uses soft-min for overlap constraints to improve numerical stability.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = circles.flatten()

    # Objective: maximize sum of radii => minimize negative sum
    def objective(x):
        r = x[2::3]
        return -np.sum(r)

    # Constraints
    constraints = []

    # Boundary constraints
    for i in range(n):
        # x >= r  => x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # x + r <= w => w - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        # y >= r  => y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # y + r <= h => h - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        # r > 0 => r >= 1e-6
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    # Overlap constraints using smooth surrogate (soft-min)
    # Instead of hard constraint dist >= r_i + r_j, use:
    # dist - r_i - r_j + eps * log(1 + exp(-k*(dist - r_i - r_j))) >= 0
    # This is a smooth approximation that's easier for SLSQP
    k = 50.0  # Sharpness of soft-min
    eps = 1e-4  # Small regularization
    
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_constraint_smooth(x, i=i, j=j):
                dx = x[i*3] - x[j*3]
                dy = x[i*3+1] - x[j*3+1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                r_sum = x[i*3+2] + x[j*3+2]
                margin = dist - r_sum
                # Soft constraint: always >= 0, smooth
                return margin + eps * np.log(1.0 + np.exp(-k * margin))
            
            constraints.append({
                'type': 'ineq',
                'fun': overlap_constraint_smooth
            })

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False}
        )

        if res.success or res.fun < -np.sum(circles[:, 2]) + 1e-6:
            new_circles = res.x.reshape(-1, 3)
            # Validate and fix any tiny numerical issues
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                # Try to fix with relaxation
                fixed = _resolve_collisions(new_circles, w, h, max_iters=200)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass

    return None

def _polish_pass(circles, w, h, deadline):
    """
    High-precision polish: local optimization with tighter tolerances.
    Uses coordinate descent on radii with position re-optimization.
    """
    if time.time() > deadline - 3.0:
        return circles
    
    n = len(circles)
    best = circles.copy()
    best_score = np.sum(best[:, 2])
    
    # Try small perturbations to each radius
    for i in range(n):
        if time.time() > deadline - 2.0:
            break
            
        orig_r = best[i, 2]
        
        # Try increasing radius slightly
        for delta in [1e-4, 5e-5, 1e-5]:
            test = best.copy()
            test[i, 2] = orig_r + delta
            
            # Re-optimize positions for this fixed radius configuration
            # Use a quick SLSQP with only position variables
            def obj_pos(x):
                # x contains only positions [x0,y0,x1,y1,...]
                pos = x.reshape(n, 2)
                # Check if valid
                valid = True
                for j in range(n):
                    if pos[j, 0] < test[j, 2] or pos[j, 0] > w - test[j, 2]:
                        valid = False
                        break
                    if pos[j, 1] < test[j, 2] or pos[j, 1] > h - test[j, 2]:
                        valid = False
                        break
                if not valid:
                    return 1e6
                
                for j in range(n):
                    for k in range(j+1, n):
                        dx = pos[j, 0] - pos[k, 0]
                        dy = pos[j, 1] - pos[k, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < test[j, 2] + test[k, 2] - 1e-8:
                            return 1e6 + (test[j, 2] + test[k, 2] - dist)
                
                return 0.0
            
            x0_pos = best[:, :2].flatten()
            
            bounds_pos = []
            for j in range(n):
                bounds_pos.append((test[j, 2], w - test[j, 2]))
                bounds_pos.append((test[j, 2], h - test[j, 2]))
            
            try:
                res = minimize(obj_pos, x0_pos, method='SLSQP', 
                              bounds=bounds_pos, options={'maxiter': 50, 'ftol': 1e-10})
                
                if res.fun < 1e5:  # Valid solution
                    new_pos = res.x.reshape(n, 2)
                    test_circles = np.column_stack((new_pos, test[:, 2]))
                    
                    if _validate_solution(test_circles, w, h, N):
                        new_score = np.sum(test_circles[:, 2])
                        if new_score > best_score:
                            best = test_circles
                            best_score = new_score
            except:
                pass
        
        # Also try decreasing radius slightly (might allow others to grow)
        for delta in [1e-4, 5e-5]:
            test = best.copy()
            test[i, 2] = max(orig_r - delta, 1e-6)
            
            # Quick check if this allows overall improvement
            relaxed = _resolve_collisions(test, w, h, max_iters=50)
            if _validate_solution(relaxed, w, h, N):
                score = np.sum(relaxed[:, 2])
                if score > best_score:
                    best = relaxed
                    best_score = score
    
    return best

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio.
    1. Generate multiple initializations (grid + random).
    2. Force-directed relaxation.
    3. SLSQP optimize with smooth constraints.
    4. Polish pass for high precision.
    """
    best_result = None
    best_score = 0.0
    
    # Generate multiple initializations
    initializations = []
    
    # Grid configurations
    ar = w / h
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6), (5, 5)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4), (5, 5)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4)]
        
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        initializations.append(init_circles)
    
    # Add random initialization
    np.random.seed(42)
    random_circles = np.zeros((N, 3))
    for i in range(N):
        r = 0.05 * min(w, h)
        random_circles[i, 0] = np.random.uniform(r, w - r)
        random_circles[i, 1] = np.random.uniform(r, h - r)
        random_circles[i, 2] = r
    initializations.append(random_circles)
    
    # Process each initialization
    for init in initializations:
        if time.time() > deadline - 15.0:
            break
            
        # Force-directed relaxation
        relaxed_fd = _force_directed_pack(init, w, h, max_iters=300)
        
        # Standard relaxation
        relaxed_std = _resolve_collisions(init, w, h, max_iters=200)
        
        # Use the better of the two
        if _validate_solution(relaxed_fd, w, h, N) and _validate_solution(relaxed_std, w, h, N):
            if np.sum(relaxed_fd[:, 2]) >= np.sum(relaxed_std[:, 2]):
                best_init = relaxed_fd
            else:
                best_init = relaxed_std
        elif _validate_solution(relaxed_fd, w, h, N):
            best_init = relaxed_fd
        elif _validate_solution(relaxed_std, w, h, N):
            best_init = relaxed_std
        else:
            continue
        
        # SLSQP optimization
        if time.time() < deadline - 10.0:
            optimized = _slsqp_optimize(best_init, w, h, deadline)
            if optimized is not None and _validate_solution(optimized, w, h, N):
                best_init = optimized
        
        # Polish pass
        if time.time() < deadline - 5.0:
            polished = _polish_pass(best_init, w, h, deadline)
            if _validate_solution(polished, w, h, N):
                best_init = polished
        
        score = np.sum(best_init[:, 2])
        if score > best_score:
            best_score = score
            best_result = best_init.copy()
    
    return best_result

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 115.0 # Safety margin under 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try - focus on promising ranges
    # For 21 circles, square and slightly elongated are likely optimal
    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75, 1.1, 0.95, 1.4, 0.7, 1.5, 0.65]
    
    for ar in aspect_ratios:
        if time.time() > deadline_global - 20.0:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        result = _solve_aspect(w, h, deadline_global)
        
        if result is not None and _validate_solution(result, w, h, N):
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result.copy()

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _generate_grid_init(w, h, N, 5, 5)
        best_result = _resolve_collisions(best_result, w, h, max_iters=200)
        if not _validate_solution(best_result, w, h, N):
            for i in range(N):
                x, y, r = best_result[i]
                min_d = min(x, w-x, y, h-y)
                best_result[i, 2] = min(r, max(min_d - 1e-4, 1e-6))
            best_result = _resolve_collisions(best_result, w, h, max_iters=200)

    return best_result
