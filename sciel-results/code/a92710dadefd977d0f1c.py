import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(circles):
    """Check if circles are valid: within a perimeter-4 rectangle, non-overlapping."""
    if circles.shape[0] != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if 2 * (w + h) > 4.0 + 1e-9:
        return False
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    
    return True

def _total_radius(circles):
    return np.sum(circles[:, 2])

def _make_grid(n, w, h):
    """Create a grid of n circles in a w x h rectangle."""
    # Find grid dimensions
    best = None
    for rows in range(1, n + 1):
        cols = (n + rows - 1) // rows
        if rows * cols >= n:
            area = rows * cols
            if best is None or area < best[0]:
                best = (area, rows, cols)
    if best is None:
        rows, cols = n, 1
    else:
        _, rows, cols = best
    
    circles = np.zeros((n, 3))
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) * w / cols
        y = (row + 0.5) * h / rows
        r = min(w / cols, h / rows) * 0.45
        circles[i] = [x, y, r]
    return circles

def _relax_circles(circles, w, h, iterations=200):
    """Relax circles to remove overlaps and keep within bounds."""
    circles = circles.copy()
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist - 1e-10:
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    overlap = min_dist - dist
                    ux, uy = dx / dist, dy / dist
                    push = overlap / 2 + 1e-6
                    circles[i, 0] += ux * push
                    circles[i, 1] += uy * push
                    circles[j, 0] -= ux * push
                    circles[j, 1] -= uy * push
                    moved = True
        
        # Keep within bounds
        for i in range(N):
            r = circles[i, 2]
            circles[i, 0] = np.clip(circles[i, 0], r, w - r)
            circles[i, 1] = np.clip(circles[i, 1], r, h - r)
            if circles[i, 0] < r or circles[i, 0] > w - r or circles[i, 1] < r or circles[i, 1] > h - r:
                moved = True
        
        if not moved:
            break
    return circles

def _slsqp_optimize(circles, w, h, max_iter=100, deadline=None):
    """Optimize circles using SLSQP with explicit constraints."""
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    def objective(x):
        return -np.sum(x[2 * N:])
    
    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[2 * N:] = -1.0
        return grad
    
    cons = []
    
    # Non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[i] - x[j])**2 + (x[N+i] - x[N+j])**2 - (x[2*N+i] + x[2*N+j])**2,
            })
    
    # Boundary constraints: circles must be within [0, w] x [0, h]
    for i in range(N):
        # x_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
        # x_i <= w - r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i] - x[2*N+i]})
        # y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
        # y_i <= h - r_i
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[N+i] - x[2*N+i]})
    
    bounds = [(0.001, w - 0.001)] * N + [(0.001, h - 0.001)] * N + [(0.001, 0.5)] * N
    
    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        if res.success:
            x = res.x
            circles_out = np.column_stack([x[:N], x[N:2*N], x[2*N:]])
            return circles_out
    except Exception:
        pass
    
    return circles

def _polish_slsqp(circles, w, h, deadline=None):
    """Polish using SLSQP with tighter tolerances."""
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    def objective(x):
        return -np.sum(x[2 * N:])
    
    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[2 * N:] = -1.0
        return grad
    
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[i] - x[j])**2 + (x[N+i] - x[N+j])**2 - (x[2*N+i] + x[2*N+j])**2,
            })
    
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[N+i] - x[2*N+i]})
    
    bounds = [(0.001, w - 0.001)] * N + [(0.001, h - 0.001)] * N + [(0.001, 0.5)] * N
    
    try:
        res = minimize(
            objective,
            x0,
            jac=objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if res.success:
            x = res.x
            circles_out = np.column_stack([x[:N], x[N:2*N], x[2*N:]])
            return circles_out
    except Exception:
        pass
    
    return circles

def _solve_aspect(ratio, deadline=None):
    """Solve for a given aspect ratio w/h = ratio, with 2*(w+h) <= 4."""
    # 2*(w + h) = 4 => w + h = 2
    # w = ratio * h
    # ratio * h + h = 2 => h(1 + ratio) = 2 => h = 2 / (1 + ratio)
    h = 2.0 / (1.0 + ratio)
    w = ratio * h
    
    # Make grid
    circles = _make_grid(N, w, h)
    circles = _relax_circles(circles, w, h, iterations=200)
    
    # SLSQP optimize
    circles = _slsqp_optimize(circles, w, h, max_iter=80, deadline=deadline)
    
    # Polish
    circles = _polish_slsqp(circles, w, h, deadline=deadline)
    
    return circles, w, h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_circles = None
    best_score = 0.0
    best_w = 1.0
    best_h = 1.0
    
    # Try different aspect ratios
    ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 0.9, 1.1]
    
    for ratio in ratios:
        if time.time() > deadline:
            break
        
        try:
            circles, w, h = _solve_aspect(ratio, deadline=deadline)
            
            # Validate
            if _validate(circles):
                score = _total_radius(circles)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
                    best_w = w
                    best_h = h
        except Exception:
            continue
    
    # If no valid solution found, create a simple valid one
    if best_circles is None:
        w, h = 1.0, 1.0
        circles = _make_grid(N, w, h)
        circles = _relax_circles(circles, w, h, iterations=500)
        if _validate(circles):
            best_circles = circles
        else:
            # Absolute fallback
            best_circles = np.zeros((N, 3))
            for i in range(N):
                row = i // 5
                col = i % 5
                if col == 4 and row >= 4:
                    col = 3
                best_circles[i] = [0.1 + col * 0.18, 0.1 + row * 0.18, 0.07]
            best_w, best_h = 1.0, 1.0
    
    return best_circles
