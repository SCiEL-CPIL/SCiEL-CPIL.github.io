import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-6):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _make_hexagonal_initial(w, h, rows=None):
    """
    Create a hexagonal-like arrangement for 21 circles.
    """
    if rows is None:
        # 21 circles: 5, 4, 5, 4, 3 is a good hex-like pattern
        rows = [5, 4, 5, 4, 3]
    n_rows = len(rows)
    circles = []
    
    # Estimate base radius
    # Area of rectangle is w*h. Area of 21 circles is 21 * pi * r^2.
    # Packing density ~ 0.9 for hex.
    r_est = np.sqrt(w * h / (21 * np.pi * 0.9))
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / n_rows
        for col_j in range(count):
            x = (col_j + 0.5) * w / count
            r = r_est
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _relax_circles_safe(circles, w, h, iterations=1000, tol=1e-9):
    """
    Relaxation to ensure validity.
    """
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for it in range(iterations):
        moved = False
        
        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] < rad[i]:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] > w - rad[i]:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] < rad[i]:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] > h - rad[i]:
                pos[i, 1] = h - rad[i]
                moved = True
        
        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True
        
        if not moved:
            break
    
    new_circles = np.column_stack((pos, rad))
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    return circles

def _penalty_objective(x, w, h, beta, eps=1e-4):
    """
    Penalty based objective for L-BFGS-B.
    x = [x0..xN, y0..yN, r0..rN]
    """
    n = N
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    obj = -np.sum(r)
    
    def smooth_max(v):
        return 0.5 * (v + np.sqrt(v*v + eps*eps))
    
    # Boundary violations
    v_left = r - cx
    v_right = cx + r - w
    v_bottom = r - cy
    v_top = cy + r - h
    
    obj += beta * (np.sum(smooth_max(v_left)**2) + 
                   np.sum(smooth_max(v_right)**2) + 
                   np.sum(smooth_max(v_bottom)**2) + 
                   np.sum(smooth_max(v_top)**2))
    
    # Overlap penalties
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    dist = np.sqrt(dx*dx + dy*dy)
    np.fill_diagonal(dist, 1e12)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    v_overlap = r_sum - dist
    
    iu, ju = np.triu_indices(n, k=1)
    v_vec = v_overlap[iu, ju]
    obj += beta * np.sum(smooth_max(v_vec)**2)
    
    return obj

def _solve_penalty(circles, w, h, deadline, initial_beta=100.0, max_iters=200):
    """
    Optimize using L-BFGS-B with penalty method.
    """
    if time.time() > deadline - 10.0:
        return circles

    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))
        
    best_circles = circles.copy()
    best_val = _total_radius(circles)
    
    # Annealing penalty weight
    betas = [initial_beta, initial_beta * 5, initial_beta * 25, initial_beta * 125]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                _penalty_objective,
                current_x,
                args=(w, h, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_x = res.x
        except Exception:
            continue
            
        new_circles = np.column_stack((current_x[:n], current_x[n:2*n], current_x[2*n:]))
        
        # Relax to ensure validity
        new_circles = _relax_circles_safe(new_circles, w, h, iterations=200)
        
        if _validate(new_circles, w, h, tol=1e-5):
            val = _total_radius(new_circles)
            if val > best_val:
                best_val = val
                best_circles = new_circles.copy()
                current_x = best_circles.flatten()
            else:
                current_x = best_circles.flatten()
                
    return best_circles

def _pack_for_width(w, deadline):
    """
    Pack circles in a rectangle of width w and height 2-w.
    Returns the best sum of radii found.
    """
    h = 2.0 - w
    if h <= 0 or w <= 0:
        return -np.inf
        
    # Initial guess
    circles = _make_hexagonal_initial(w, h)
    circles = _relax_circles_safe(circles, w, h, iterations=500)
    
    if not _validate(circles, w, h):
        # Fallback to grid if hex fails
        circles = _make_grid_initial(w, h)
        circles = _relax_circles_safe(circles, w, h, iterations=500)
        
    if not _validate(circles, w, h):
        return -np.inf
        
    val = _total_radius(circles)
    
    # Optimize
    if time.time() < deadline - 5.0:
        opt = _solve_penalty(circles, w, h, deadline)
        if opt is not None:
            circles = opt
            val = _total_radius(circles)
            
    return val, circles

def _make_grid_initial(w, h, rows=None, cols=None):
    """
    Create a grid arrangement for 21 circles.
    """
    if rows is None or cols is None:
        best_r, best_c = 3, 7
        best_diff = abs(w/h - 7/3)
        for r in range(1, 22):
            for c in range(1, 22):
                if r * c >= 21:
                    diff = abs(w/h - c/r)
                    if diff < best_diff:
                        best_diff = diff
                        best_r, best_c = r, c
        rows, cols = best_r, best_c
        
    circles = []
    r_est = min(w / cols, h / rows) / 2
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= N:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            circles.append([x, y, r_est])
            
    return np.array(circles[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    best_w = 1.0
    
    # Use Nelder-Mead to find the best aspect ratio (width)
    # We wrap the inner solver to return a scalar (negative sum of radii)
    
    def nm_objective(w_vec):
        w = w_vec[0]
        # Keep w within valid bounds (0, 2)
        if w <= 0.1 or w >= 1.9:
            return 1e6
        val, circles = _pack_for_width(w, deadline)
        if val == -np.inf:
            return 1e6
        # Store the best circles found
        nonlocal best_result, best_value, best_w
        if val > best_value:
            best_value = val
            best_result = circles
            best_w = w
        return -val  # Minimize negative sum of radii

    # Initial guess for Nelder-Mead: Square (w=1)
    x0 = np.array([1.0])
    
    try:
        # Nelder-Mead options
        nm_options = {
            'xatol': 1e-4,
            'fatol': 1e-6,
            'maxiter': 100,
            'adaptive': True
        }
        
        res = minimize(nm_objective, x0, method='Nelder-Mead', options=nm_options)
        
    except Exception as e:
        print(f"Nelder-Mead failed: {e}")
        
    # If Nelder-Mead didn't find anything, try a few manual points
    if best_result is None:
        for w in [0.8, 1.0, 1.2]:
            val, circles = _pack_for_width(w, deadline)
            if val > best_value:
                best_value = val
                best_result = circles
                best_w = w

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Best Value: {_total_radius(res)}")
