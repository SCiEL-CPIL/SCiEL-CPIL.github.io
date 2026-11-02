"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # safety margin before 120s limit


def _check_validity(arr, tol=1e-9):
    """Check if circles are valid: disjoint interiors, inside bounding box, perimeter <= 4."""
    n = arr.shape[0]
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    x_ext_min = np.min(xs - rs)
    x_ext_max = np.max(xs + rs)
    y_ext_min = np.min(ys - rs)
    y_ext_max = np.max(ys + rs)
    
    perim = 2.0 * ((x_ext_max - x_ext_min) + (y_ext_max - y_ext_min))
    if perim > 4.0 + tol:
        return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _max_radius_for_circle(arr, i, w, h, tol=1e-10):
    """Compute the maximum radius for circle i given other circles and boundaries."""
    x, y, r = arr[i, 0], arr[i, 1], arr[i, 2]
    
    # Boundary constraints
    r_max = min(x, w - x, y, h - y)
    
    # Other circles
    for j in range(N):
        if j == i:
            continue
        dx = x - arr[j, 0]
        dy = y - arr[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        # Need dist >= r + arr[j,2], so r <= dist - arr[j,2]
        r_max = min(r_max, dist - arr[j, 2])
    
    return max(r_max, 0.001)


def _expand_circles(arr, w, h, max_iter=200, tol=1e-12):
    """Iteratively expand circles to maximum radius."""
    for it in range(max_iter):
        changed = False
        for i in range(N):
            new_r = _max_radius_for_circle(arr, i, w, h)
            if new_r > arr[i, 2] + tol:
                arr[i, 2] = new_r
                changed = True
        if not changed:
            break
    return arr


def _force_relaxation(arr, w, h, max_iter=500, tol=1e-10):
    """Force-directed relaxation to separate overlapping circles."""
    for it in range(max_iter):
        moved = False
        for i in range(N):
            fx, fy = 0.0, 0.0
            for j in range(N):
                if j == i:
                    continue
                dx = arr[i, 0] - arr[j, 0]
                dy = arr[i, 1] - arr[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    dx, dy = 1e-6, 1e-6
                    dist = np.sqrt(dx * dx + dy * dy)
                overlap = arr[i, 2] + arr[j, 2] - dist
                if overlap > 0:
                    force = overlap * 0.5
                    fx += force * dx / dist
                    fy += force * dy / dist
            
            # Boundary forces
            if arr[i, 0] < arr[i, 2]:
                fx += (arr[i, 2] - arr[i, 0]) * 2.0
            if arr[i, 0] > w - arr[i, 2]:
                fx -= (arr[i, 0] - (w - arr[i, 2])) * 2.0
            if arr[i, 1] < arr[i, 2]:
                fy += (arr[i, 2] - arr[i, 1]) * 2.0
            if arr[i, 1] > h - arr[i, 2]:
                fy -= (arr[i, 1] - (h - arr[i, 2])) * 2.0
            
            arr[i, 0] += fx
            arr[i, 1] += fy
            
            # Clamp to bounds
            arr[i, 0] = np.clip(arr[i, 0], arr[i, 2], w - arr[i, 2])
            arr[i, 1] = np.clip(arr[i, 1], arr[i, 2], h - arr[i, 2])
            
            if abs(fx) + abs(fy) > tol:
                moved = True
        
        if not moved:
            break
    return arr


def _polish_result(arr, w, h):
    """Polish the result with expansion and relaxation."""
    arr = arr.copy()
    arr = _expand_circles(arr, w, h)
    arr = _force_relaxation(arr, w, h)
    arr = _expand_circles(arr, w, h)
    return arr


def _is_valid(arr, w, h, tol=1e-9):
    """Fast validity check for circles in a fixed rectangle [0,w] x [0,h]."""
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _grid_initialization(w, h, rows, cols):
    """Initialize circles on a grid pattern."""
    arr = np.zeros((N, 3))
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            r = min(w / cols, h / rows) * 0.45
            arr[count, 0] = x
            arr[count, 1] = y
            arr[count, 2] = r
            count += 1
    # Fill remaining with small circles
    while count < N:
        arr[count, 0] = random.uniform(0.1 * w, 0.9 * w)
        arr[count, 1] = random.uniform(0.1 * h, 0.9 * h)
        arr[count, 2] = 0.05
        count += 1
    return arr


def _random_initialization(w, h):
    """Random initialization."""
    arr = np.zeros((N, 3))
    for i in range(N):
        arr[i, 0] = random.uniform(0.1 * w, 0.9 * w)
        arr[i, 1] = random.uniform(0.1 * h, 0.9 * h)
        arr[i, 2] = random.uniform(0.05, 0.15)
    return arr


def _optimize_from_initial(arr, w, h, time_budget=30.0):
    """Optimize from an initial configuration."""
    start = time.time()
    
    # First, relax to remove overlaps
    arr = _force_relaxation(arr, w, h, max_iter=100)
    
    # Expand circles
    arr = _expand_circles(arr, w, h)
    
    # SLSQP refinement
    dim = 3 * N
    xs = arr[:, 0].copy()
    ys = arr[:, 1].copy()
    rs = arr[:, 2].copy()
    
    def objective(z):
        xs_z = z[0::3]
        ys_z = z[1::3]
        rs_z = z[2::3]
        
        # Penalty for violations
        penalty = 0.0
        
        # Boundary
        bound_viol = np.maximum(0, rs_z - xs_z) + np.maximum(0, rs_z - (w - xs_z))
        bound_viol += np.maximum(0, rs_z - ys_z) + np.maximum(0, rs_z - (h - ys_z))
        penalty += 100.0 * np.sum(bound_viol)
        
        # Pairwise
        for i in range(N):
            dx = xs_z[i] - xs_z[i+1:]
            dy = ys_z[i] - ys_z[i+1:]
            dists = np.sqrt(dx*dx + dy*dy)
            violations = np.maximum(0, rs_z[i] + rs_z[i+1:] - dists)
            penalty += 100.0 * np.sum(violations)
        
        penalty += 1000.0 * np.sum(np.maximum(0, -rs_z))
        
        return -(np.sum(rs_z) - penalty)
    
    def make_constraint(i, j):
        def fun(z):
            dx = z[3*i] - z[3*j]
            dy = z[3*i+1] - z[3*j+1]
            return np.sqrt(dx*dx + dy*dy) - z[3*i+2] - z[3*j+2]
        return fun
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: w - z[3*i] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: h - z[3*i+1] - z[3*i+2]})
    
    z0 = np.zeros(dim)
    for i in range(N):
        z0[3*i] = xs[i]
        z0[3*i+1] = ys[i]
        z0[3*i+2] = rs[i]
    
    lb = np.zeros(dim)
    ub = np.zeros(dim)
    for i in range(N):
        lb[3*i] = 0.0
        ub[3*i] = w
        lb[3*i+1] = 0.0
        ub[3*i+1] = h
        lb[3*i+2] = 0.01
        ub[3*i+2] = min(w, h) / 2.0
    
    try:
        result = minimize(
            objective,
            z0,
            method='SLSQP',
            bounds=list(zip(lb, ub)),
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-10}
        )
        
        if time.time() - start < time_budget and result.success:
            arr_new = np.zeros((N, 3))
            for i in range(N):
                arr_new[i, 0] = result.x[3*i]
                arr_new[i, 1] = result.x[3*i+1]
                arr_new[i, 2] = result.x[3*i+2]
            
            # Clamp
            for i in range(N):
                arr_new[i, 0] = np.clip(arr_new[i, 0], arr_new[i, 2], w - arr_new[i, 2])
                arr_new[i, 1] = np.clip(arr_new[i, 1], arr_new[i, 2], h - arr_new[i, 2])
            
            if _is_valid(arr_new, w, h):
                arr = arr_new
    except Exception:
        pass
    
    # Final expansion
    arr = _expand_circles(arr, w, h)
    arr = _force_relaxation(arr, w, h, max_iter=50)
    arr = _expand_circles(arr, w, h)
    
    return arr


def construct():
    start_time = time.time()
    
    best_arr = None
    best_sum = 0.0
    
    # Try different aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9, 1.4, 0.7]
    
    for aspect in aspects:
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Try multiple initializations
        inits = []
        
        # Grid initializations with different layouts
        inits.append(_grid_initialization(w, h, 3, 7))
        inits.append(_grid_initialization(w, h, 4, 6))
        inits.append(_grid_initialization(w, h, 5, 5))
        inits.append(_grid_initialization(w, h, 3, 8))
        
        # Random initializations
        for _ in range(3):
            inits.append(_random_initialization(w, h))
        
        for init_arr in inits:
            if time.time() - start_time > DEADLINE - 5.0:
                break
            
            time_budget = min(15.0, DEADLINE - (time.time() - start_time) - 5.0)
            if time_budget <= 2.0:
                break
            
            arr = _optimize_from_initial(init_arr, w, h, time_budget=time_budget)
            
            if _is_valid(arr, w, h):
                s = np.sum(arr[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_arr = arr.copy()
    
    # Final fallback
    if best_arr is None or not _check_validity(best_arr):
        w = h = 1.0
        side = 5
        best_arr = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                best_arr[count, 0] = (i + 0.5) / side
                best_arr[count, 1] = (j + 0.5) / side
                best_arr[count, 2] = 0.5 / side - 0.01
                count += 1
    
    return best_arr
