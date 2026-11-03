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

def _make_grid(n, w, h, cols, rows):
    """Create a grid of n circles with computed radii that fit."""
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5) / cols * w
            cy = (i + 0.5) / rows * h
            # Radius limited by spacing and boundaries
            dx = w / cols
            dy = h / rows
            r = 0.5 * min(dx, dy) * 0.99
            # Also check boundary constraints
            r = min(r, cx, w - cx, cy, h - cy)
            r = max(r, 0.001)
            circles.append((cx, cy, r))
    circles = circles[:n]
    while len(circles) < n:
        circles.append((w / 2, h / 2, 0.001))
    return circles

def _relax_circles(circles, w, h, iterations=2000, tol=1e-10):
    """
    Iterative relaxation: separate overlapping circles and push back from walls.
    This is a simple but effective method to get a feasible starting point.
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
                    # Random push
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

def _slsqp_optimize(circles, w, h, deadline):
    """
    SLSQP optimization with explicit constraints.
    Variables: [cx_1, cy_1, r_1, cx_2, cy_2, r_2, ...]
    Maximize: sum of radii
    Constraints:
      - cx_i >= r_i, cx_i <= w - r_i
      - cy_i >= r_i, cy_i <= h - r_i
      - r_i >= 0
      - dist(i,j) >= r_i + r_j
    """
    n = len(circles)
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
    # Boundary constraints: cx - r >= 0
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2]})
    
    # Non-overlap constraints
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
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.success or res.fun < 0:
            best_circles = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)]
            if _validate(best_circles, w, h, tol=1e-8):
                return best_circles
    except Exception:
        pass
    
    return None

def _polish_slsqp(circles, w, h, deadline):
    """Final polish with SLSQP for higher precision."""
    if time.time() > deadline - 3.0:
        return circles
    
    n = len(circles)
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
            options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
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
    
    # Aspect ratios: w + h = 2
    aspect_ratios = [
        (1.0, 1.0),
        (0.9, 1.1),
        (0.8, 1.2),
        (0.7, 1.3),
        (1.1, 0.9),
        (1.2, 0.8),
        (1.3, 0.7),
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
    ]
    
    for w, h in aspect_ratios:
        if time.time() > _deadline - 15.0:
            break
        
        for cols, rows in grid_configs:
            if time.time() > _deadline - 10.0:
                break
            
            # Create initial grid
            circles = _make_grid(N, w, h, cols, rows)
            
            # Relax to resolve overlaps
            relaxed = _relax_circles(circles, w, h, iterations=500)
            
            if not _validate(relaxed, w, h, tol=1e-6):
                continue
            
            score = _total_radius(relaxed)
            if score > best_score:
                best_score = score
                best_circles = relaxed
            
            # SLSQP optimization
            if time.time() < _deadline - 10.0:
                optimized = _slsqp_optimize(relaxed, w, h, _deadline)
                if optimized is not None:
                    opt_score = _total_radius(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_circles = optimized
    
    # Final polish
    if best_circles is not None and time.time() < _deadline - 3.0:
        for w, h in [(1.0, 1.0), (0.9, 1.1), (0.8, 1.2)]:
            if time.time() > _deadline - 2.0:
                break
            polished = _polish_slsqp(best_circles, w, h, _deadline)
            pol_score = _total_radius(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_circles = polished
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback: 5x5 grid in unit square
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
