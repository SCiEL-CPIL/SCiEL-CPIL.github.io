"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _make_hexagonal_grid(n, w, h, cols, rows):
    """Create a hexagonal-like grid of n circles."""
    circles = []
    for i in range(rows):
        offset = (i % 2) * 0.5
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5 + offset) / cols * w
            cy = (i + 0.5) / rows * h
            dx = w / cols
            dy = h / rows
            r = 0.5 * min(dx, dy) * 0.95
            r = min(r, cx, w - cx, cy, h - cy)
            r = max(r, 0.001)
            circles.append((cx, cy, r))
    circles = circles[:n]
    while len(circles) < n:
        circles.append((w / 2, h / 2, 0.001))
    return circles

def _make_grid(n, w, h, cols, rows):
    """Create a grid of n circles with computed radii that fit."""
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5) / cols * w
            cy = (i + 0.5) / rows * h
            dx = w / cols
            dy = h / rows
            r = 0.5 * min(dx, dy) * 0.99
            r = min(r, cx, w - cx, cy, h - cy)
            r = max(r, 0.001)
            circles.append((cx, cy, r))
    circles = circles[:n]
    while len(circles) < n:
        circles.append((w / 2, h / 2, 0.001))
    return circles

def _relax_circles(circles, w, h, iterations=3000, tol=1e-12):
    """
    Iterative relaxation: separate overlapping circles and push back from walls.
    """
    n = len(circles)
    circles = [list(c) for c in circles]
    
    for it in range(iterations):
        if time.time() > _deadline - 5.0:
            break
        max_move = 0.0
        for i in range(n):
            cx, cy, r = circles[i]
            fx, fy = 0.0, 0.0
            # Boundary forces
            if cx - r < 0:
                fx += (r - cx) * 2.0
            if cx + r > w:
                fx -= (cx + r - w) * 2.0
            if cy - r < 0:
                fy += (r - cy) * 2.0
            if cy + r > h:
                fy -= (cy + r - h) * 2.0
            # Overlap forces
            for j in range(n):
                if i == j:
                    continue
                dx = cx - circles[j][0]
                dy = cy - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r + circles[j][2]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    fx += dx / dist * overlap
                    fy += dy / dist * overlap
                elif dist <= 1e-12:
                    angle = np.random.uniform(0, 2 * np.pi)
                    fx += np.cos(angle) * min_dist
                    fy += np.sin(angle) * min_dist
            # Apply force with damping
            step = 0.5
            new_cx = cx + step * fx
            new_cy = cy + step * fy
            max_move = max(max_move, abs(new_cx - cx), abs(new_cy - cy))
            circles[i][0] = new_cx
            circles[i][1] = new_cy
        if max_move < tol:
            break
    
    # Final clamp to boundaries
    for i in range(n):
        cx, cy, r = circles[i]
        cx = max(r, min(w - r, cx))
        cy = max(r, min(h - r, cy))
        circles[i][0] = cx
        circles[i][1] = cy
    
    return [tuple(c) for c in circles]

def _smooth_overlap_penalty(x, w, h, eps):
    """
    Compute smooth overlap penalty using log-sum-exp for distance constraints.
    x: (3N,) array [cx1, cy1, r1, cx2, cy2, r2, ...]
    Returns penalty value and gradient.
    """
    n = N
    penalty = 0.0
    grad = np.zeros(3 * n)
    
    # Boundary penalties using smooth max
    def _smooth_max(val, eps):
        """Smooth approximation of max(0, val) using log-sum-exp."""
        if val > eps:
            return val
        elif val < -eps:
            return 0.0
        else:
            # Smooth transition
            return eps * np.log(1.0 + np.exp(val / eps))
    
    def _smooth_max_grad(val, eps):
        """Gradient of smooth max."""
        if val > eps:
            return 1.0
        elif val < -eps:
            return 0.0
        else:
            return 1.0 / (1.0 + np.exp(-val / eps))
    
    # Boundary penalties
    for i in range(n):
        cx, cy, r = x[i*3], x[i*3+1], x[i*3+2]
        # Left boundary: constraint is cx - r >= 0
        viol = r - cx
        if abs(viol) < 10 * eps or viol > 0:
            sm = _smooth_max(viol, eps)
            penalty += sm * sm
            g_sm = _smooth_max_grad(viol, eps)
            grad[i*3] += 2 * sm * g_sm
            grad[i*3+2] -= 2 * sm * g_sm
        # Right boundary
        viol = cx + r - w
        if abs(viol) < 10 * eps or viol > 0:
            sm = _smooth_max(viol, eps)
            penalty += sm * sm
            g_sm = _smooth_max_grad(viol, eps)
            grad[i*3] += 2 * sm * g_sm
            grad[i*3+2] += 2 * sm * g_sm
        # Bottom boundary
        viol = r - cy
        if abs(viol) < 10 * eps or viol > 0:
            sm = _smooth_max(viol, eps)
            penalty += sm * sm
            g_sm = _smooth_max_grad(viol, eps)
            grad[i*3+1] += 2 * sm * g_sm
            grad[i*3+2] -= 2 * sm * g_sm
        # Top boundary
        viol = cy + r - h
        if abs(viol) < 10 * eps or viol > 0:
            sm = _smooth_max(viol, eps)
            penalty += sm * sm
            g_sm = _smooth_max_grad(viol, eps)
            grad[i*3+1] += 2 * sm * g_sm
            grad[i*3+2] += 2 * sm * g_sm
    
    # Overlap penalties with log-sum-exp smooth max
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i*3] - x[j*3]
            dy = x[i*3+1] - x[j*3+1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-14 else 1e-7
            min_dist = x[i*3+2] + x[j*3+2]
            overlap = min_dist - dist
            if abs(overlap) < 10 * eps or overlap > 0:
                sm = _smooth_max(overlap, eps)
                penalty += sm * sm
                g_sm = _smooth_max_grad(overlap, eps)
                if dist > 1e-12:
                    grad[i*3] += 2 * sm * g_sm * dx / dist
                    grad[i*3+1] += 2 * sm * g_sm * dy / dist
                    grad[j*3] -= 2 * sm * g_sm * dx / dist
                    grad[j*3+1] -= 2 * sm * g_sm * dy / dist
                grad[i*3+2] += 2 * sm * g_sm
                grad[j*3+2] += 2 * sm * g_sm
    
    return penalty, grad

def _penalty_optimize(circles, w, h, deadline, max_iter=500):
    """
    Penalty-based optimization using L-BFGS-B with log-sum-exp smoothing.
    Maximize sum of radii subject to constraints via penalty.
    """
    if time.time() > deadline - 10.0:
        return circles
    
    n = N
    x0 = np.array([v for c in circles for v in c])
    
    # Progressive penalty weight with finer eps schedule
    penalty_weights = [5.0, 20.0, 100.0, 500.0, 2000.0]
    eps_values = [0.05, 0.01, 0.001, 0.0001, 0.00001]
    
    best_x = x0.copy()
    best_obj = -np.sum(x0[2::3])
    
    for pi, (pw, eps) in enumerate(zip(penalty_weights, eps_values)):
        if time.time() > deadline - 5.0:
            break
        
        def obj(x):
            if time.time() > deadline:
                return 0.0
            radii_sum = np.sum(x[2::3])
            penalty, _ = _smooth_overlap_penalty(x, w, h, eps)
            return -radii_sum + pw * penalty
        
        def grad(x):
            if time.time() > deadline:
                return np.zeros_like(x)
            radii_sum = np.sum(x[2::3])
            penalty, pgrad = _smooth_overlap_penalty(x, w, h, eps)
            g = -np.ones_like(x)
            g[2::3] = 1.0
            g += pw * pgrad
            return g
        
        bounds = [(0, w)] * n + [(0, h)] * n + [(0.001, 1.0)] * n
        
        try:
            res = minimize(
                obj,
                best_x,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Extract circles
    best_circles = [(best_x[i*3], best_x[i*3+1], best_x[i*3+2]) for i in range(n)]
    
    # Validate
    if _validate(best_circles, w, h, tol=1e-8):
        return best_circles
    
    # If invalid, try to fix by reducing radii slightly
    for i in range(n):
        best_circles[i] = (best_circles[i][0], best_circles[i][1], max(0.001, best_circles[i][2] * 0.99))
    
    if _validate(best_circles, w, h, tol=1e-6):
        return best_circles
    
    return circles

def _high_precision_polish(circles, w, h, deadline):
    """
    Final high-precision polish with progressive smooth-max epsilon
    and increasing penalty weight using log-sum-exp.
    """
    if time.time() > deadline - 3.0:
        return circles
    
    n = N
    x0 = np.array([v for c in circles for v in c])
    
    # Progressive eps from 0.001 to 1e-7 with higher penalty weights
    eps_schedule = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    penalty_weights = [200.0, 1000.0, 5000.0, 20000.0, 100000.0]
    
    best_x = x0.copy()
    best_obj = -np.sum(x0[2::3])
    
    for eps, pw in zip(eps_schedule, penalty_weights):
        if time.time() > deadline - 2.0:
            break
        
        def obj(x):
            if time.time() > deadline:
                return 0.0
            radii_sum = np.sum(x[2::3])
            penalty, _ = _smooth_overlap_penalty(x, w, h, eps)
            return -radii_sum + pw * penalty
        
        def grad(x):
            if time.time() > deadline:
                return np.zeros_like(x)
            penalty, pgrad = _smooth_overlap_penalty(x, w, h, eps)
            g = -np.ones_like(x)
            g[2::3] = 1.0
            g += pw * pgrad
            return g
        
        bounds = [(0, w)] * n + [(0, h)] * n + [(0.001, 1.0)] * n
        
        try:
            res = minimize(
                obj,
                best_x,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_circles = [(best_x[i*3], best_x[i*3+1], best_x[i*3+2]) for i in range(n)]
    
    if _validate(best_circles, w, h, tol=1e-9):
        return best_circles
    
    return circles

def _slsqp_polish(circles, w, h, deadline):
    """
    SLSQP polish with explicit constraints for final refinement.
    """
    if time.time() > deadline - 3.0:
        return circles
    
    n = N
    x0 = np.array([v for c in circles for v in c])
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)
    
    def objective_grad(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        return grad
    
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_constraint(x, i=i, j=j):
                dx = x[i*3] - x[j*3]
                dy = x[i*3+1] - x[j*3+1]
                return np.sqrt(dx*dx + dy*dy) - x[i*3+2] - x[j*3+2]
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    bounds = [(0, w)] * n + [(0, h)] * n + [(0.001, 1.0)] * n
    
    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            args=(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
        )
        
        if res.success or res.fun < 0:
            best_circles = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)]
            if _validate(best_circles, w, h, tol=1e-9):
                return best_circles
    except Exception:
        pass
    
    return circles

_deadline = 0.0

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_circles = None
    best_score = 0.0
    best_w = 1.0
    best_h = 1.0
    
    # Aspect ratios: w + h = 2, more diverse exploration
    aspect_ratios = [
        (1.0, 1.0),
        (0.95, 1.05),
        (0.9, 1.1),
        (1.05, 0.95),
        (1.1, 0.9),
        (0.85, 1.15),
        (1.15, 0.85),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.75, 1.25),
        (1.25, 0.75),
    ]
    
    # Grid configurations
    grid_configs = [
        (5, 5),
        (4, 6),
        (6, 4),
        (5, 4),
        (4, 5),
        (3, 7),
        (7, 3),
        (4, 4),
        (6, 5),
        (5, 6),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > _deadline - 20.0:
            break
        
        for cols, rows in grid_configs:
            if time.time() > _deadline - 15.0:
                break
            
            # Try both grid and hexagonal initializations
            for init_func in [_make_grid, _make_hexagonal_grid]:
                if time.time() > _deadline - 12.0:
                    break
                
                circles = init_func(N, w, h, cols, rows)
                
                # Relax to resolve overlaps
                relaxed = _relax_circles(circles, w, h, iterations=1500)
                
                if not _validate(relaxed, w, h, tol=1e-6):
                    continue
                
                score = _total_radius(relaxed)
                if score > best_score:
                    best_score = score
                    best_circles = relaxed
                    best_w = w
                    best_h = h
                
                # Penalty optimization
                if time.time() < _deadline - 10.0:
                    optimized = _penalty_optimize(relaxed, w, h, _deadline)
                    if optimized is not None:
                        opt_score = _total_radius(optimized)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_circles = optimized
                            best_w = w
                            best_h = h
                
                # Additional local polish if we have time
                if best_circles is not None and time.time() < _deadline - 8.0:
                    polished = _high_precision_polish(best_circles, best_w, best_h, _deadline)
                    pol_score = _total_radius(polished)
                    if pol_score > best_score:
                        best_score = pol_score
                        best_circles = polished
    
    # Final high-precision polish
    if best_circles is not None and time.time() < _deadline - 5.0:
        polished = _high_precision_polish(best_circles, best_w, best_h, _deadline)
        pol_score = _total_radius(polished)
        if pol_score > best_score:
            best_score = pol_score
            best_circles = polished
    
    # SLSQP polish
    if best_circles is not None and time.time() < _deadline - 2.0:
        slsqp_polished = _slsqp_polish(best_circles, best_w, best_h, _deadline)
        slsqp_score = _total_radius(slsqp_polished)
        if slsqp_score > best_score:
            best_score = slsqp_score
            best_circles = slsqp_polished
    
    # One more final polish pass if we have time
    if best_circles is not None and time.time() < _deadline - 1.0:
        final_polished = _high_precision_polish(best_circles, best_w, best_h, _deadline)
        final_score = _total_radius(final_polished)
        if final_score > best_score:
            best_score = final_score
            best_circles = final_polished
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback: 5x5 grid in unit square
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
