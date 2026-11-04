"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of their radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
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
    Default rows: 5, 4, 5, 4, 3
    """
    if rows is None:
        rows = [5, 4, 5, 4, 3]
    n_rows = len(rows)
    circles = []
    
    # Estimate base radius
    r_est = np.sqrt(w * h / (42 * np.sqrt(3)))
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / n_rows
        for col_j in range(count):
            x = (col_j + 0.5) * w / count
            r = r_est
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _make_grid_initial(w, h, rows=None, cols=None):
    """
    Create a grid arrangement for 21 circles.
    """
    if rows is None or cols is None:
        # Find a factorization close to square or matching aspect ratio
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

def _get_contact_pairs(rows):
    """
    Generate contact pairs for a hexagonal-like arrangement.
    Returns a list of (i, j) tuples.
    """
    pairs = set()
    idx = 0
    for row_i, count in enumerate(rows):
        for col_j in range(count):
            # Horizontal neighbors
            if col_j < count - 1:
                pairs.add((idx, idx + 1))
            
            # Vertical/Diagonal neighbors (row below)
            if row_i + 1 < len(rows):
                below_count = rows[row_i + 1]
                if col_j < below_count:
                    pairs.add((idx, idx + count + col_j))
                if col_j + 1 < below_count:
                    pairs.add((idx, idx + count + col_j + 1))
            
            idx += count
    
    return list(pairs)

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

def _penalty_objective(x, w, h, beta):
    """
    Penalty based objective for L-BFGS-B.
    x = [x0..xN, y0..yN, r0..rN]
    """
    n = N
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    obj = -np.sum(r)
    
    # Boundary penalties
    # Use smooth max to avoid non-differentiability at 0
    # max(0, v)^2 approximated by 0.5*(v + sqrt(v^2 + eps^2))^2 if v>0 else 0
    # Simpler: just use quadratic penalty on violations, L-BFGS-B can handle kinks often if not too sharp, 
    # but smooth is better.
    # Let's use a simple smooth penalty: 0.5 * (max(0, v))^2 is not smooth.
    # Use: 0.5 * (v + sqrt(v^2 + eps^2))^2 for v>0? No.
    # Standard smooth max: m(v) = 0.5 * (v + sqrt(v^2 + eps^2))
    # Penalty = m(v)^2
    
    eps = 1e-4
    def smooth_max(v):
        return 0.5 * (v + np.sqrt(v*v + eps*eps))
    
    v_left = r - cx
    v_right = cx + r - w
    v_bottom = r - cy
    v_top = cy + r - h
    
    obj += beta * (np.sum(smooth_max(v_left)**2) + 
                   np.sum(smooth_max(v_right)**2) + 
                   np.sum(smooth_max(v_bottom)**2) + 
                   np.sum(smooth_max(v_top)**2))
    
    # Overlap penalties
    # Vectorized
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    dist = np.sqrt(dx*dx + dy*dy)
    np.fill_diagonal(dist, 1e12)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    v_overlap = r_sum - dist
    
    # Sum of smooth_max(v)^2 for upper triangle
    iu, ju = np.triu_indices(n, k=1)
    v_vec = v_overlap[iu, ju]
    obj += beta * np.sum(smooth_max(v_vec)**2)
    
    return obj

def _solve_penalty(circles, w, h, deadline, initial_beta=100.0):
    """
    Optimize using L-BFGS-B with penalty method.
    """
    if time.time() > deadline - 10.0:
        return None

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
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
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
                # If it got worse or invalid, keep current_x as best_circles.flatten() 
                # but maybe the optimization went to a bad local min.
                # Revert to best known valid point for next iteration?
                current_x = best_circles.flatten()
                
    return best_circles

def _try_aspect(ar, deadline, rows=None, use_grid=False):
    """Try a single aspect ratio and return best result."""
    w = ar / (1 + ar) * 2
    h = 1 / (1 + ar) * 2
    
    best_val = 0.0
    best_circles = None
    
    for (ww, hh) in [(w, h), (h, w)]:
        if time.time() > deadline - 10.0:
            break
        
        try:
            if use_grid:
                circles = _make_grid_initial(ww, hh)
            else:
                circles = _make_hexagonal_initial(ww, hh, rows)
            
            circles = _relax_circles_safe(circles, ww, hh, iterations=500)
            
            if not _validate(circles, ww, hh):
                continue
            
            val = _total_radius(circles)
            
            # Try penalty-based optimization
            if time.time() < deadline - 15.0:
                opt = _solve_penalty(circles, ww, hh, deadline)
                if opt is not None:
                    circles = opt
                    val = _total_radius(circles)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
                
        except Exception:
            continue
    
    return best_circles, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Targeted aspect ratios
    ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.3, 0.8, 1.4, 0.75]
    
    # Different row configurations to explore different topologies
    row_configs = [
        [5, 4, 5, 4, 3],
        [6, 5, 5, 5],
        [7, 7, 7]
    ]
    
    # We will alternate between hex and grid inits, or just try both for a few ratios
    # To save time, let's prioritize hex for square-ish and grid for others?
    # Or just run the loop.
    
    for ar in ratios:
        if time.time() > deadline - 12.0:
            break
        
        # Try hex configurations
        for rows in row_configs:
            if time.time() > deadline - 12.0:
                break
            circles, val = _try_aspect(ar, deadline, rows, use_grid=False)
            if circles is not None and val > best_value:
                best_value = val
                best_result = circles
                
        # Try grid configuration (only for a subset to save time)
        if ar in [1.0, 1.2, 0.8]:
             if time.time() > deadline - 12.0:
                 break
             circles, val = _try_aspect(ar, deadline, use_grid=True)
             if circles is not None and val > best_value:
                 best_value = val
                 best_result = circles

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
    # Note: w/h varies, so we can't easily validate without knowing which one was best.
    # But _try_aspect returns valid circles.
