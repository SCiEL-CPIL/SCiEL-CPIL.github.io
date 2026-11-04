"""
Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Strategy:
1. Multi-start with diverse hexagonal and grid initializations.
2. Penalty-based L-BFGS-B for fast convergence from various starting points.
3. High-precision SLSQP polish with exact constraints.
4. Final L-BFGS-B refinement with very tight tolerances on the best candidate.
5. Explore a range of aspect ratios efficiently within the time budget.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # Safety margin under 150s wall-clock

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _hexagonal_init(w, h, n, variant=0):
    """
    Generate hexagonal packing initialization with variants for diversity.
    """
    # Different row configurations for 21 circles
    if variant == 0:
        rows = [4, 5, 4, 5, 3]
    elif variant == 1:
        rows = [5, 4, 5, 4, 3]
    elif variant == 2:
        rows = [4, 5, 5, 4, 3]
    elif variant == 3:
        rows = [5, 5, 4, 4, 3]
    else:
        rows = [4, 5, 4, 5, 3]
    
    total = sum(rows)
    if total < n:
        while sum(rows) < n:
            rows.append(4)
        while sum(rows) > n:
            rows[-1] -= 1
            if rows[-1] <= 0:
                rows.pop()
    elif total > n:
        while sum(rows) > n:
            rows[-1] -= 1
            if rows[-1] <= 0:
                rows.pop()
    
    circles = []
    max_circles = max(rows)
    num_rows = len(rows)
    
    # Estimate radius to fit in rectangle
    r_w = w / (2.0 * max_circles)
    r_h = h / ((num_rows - 1) * np.sqrt(3) + 2.0)
    r = min(r_w, r_h) * 0.95  # Slightly smaller to avoid initial overlaps
    
    total_width = max_circles * 2 * r
    total_height = (num_rows - 1) * r * np.sqrt(3) + 2 * r
    
    offset_x = (w - total_width) / 2.0
    offset_y = (h - total_height) / 2.0
    
    y = offset_y + r
    for i, count in enumerate(rows):
        x_start = offset_x + r
        if i % 2 == 1:
            x_start += r
        
        for j in range(count):
            cx = x_start + j * 2 * r
            cy = y
            cx = np.clip(cx, r, w - r)
            cy = np.clip(cy, r, h - r)
            circles.append([cx, cy, r])
        
        y += r * np.sqrt(3)
    
    if len(circles) != n:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = 0.5 * min(w, h) / side * 0.85
                cells.append([cx, cy, r])
        return np.array(cells[:n])
    
    return np.array(circles[:n])

def _grid_init(w, h, n, variant=0):
    """
    Generate grid initialization with variants.
    """
    side = int(np.ceil(np.sqrt(n)))
    circles = []
    
    for i in range(side):
        for j in range(side):
            if len(circles) >= n:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            
            # Add some perturbation for diversity
            if variant > 0:
                cx += np.random.uniform(-0.05, 0.05) * w / side
                cy += np.random.uniform(-0.05, 0.05) * h / side
            
            r = 0.5 * min(w / side, h / side) * 0.85
            cx = np.clip(cx, r, w - r)
            cy = np.clip(cy, r, h - r)
            circles.append([cx, cy, r])
    
    return np.array(circles[:n])

def _penalty_objective(x, w, h, penalty):
    """
    Vectorized penalty objective function.
    """
    n = len(x) // 3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    obj = -np.sum(rs)
    
    # Boundary violations
    pen_boundary = 0.0
    pen_boundary += np.sum(np.maximum(0, rs - xs))
    pen_boundary += np.sum(np.maximum(0, xs + rs - w))
    pen_boundary += np.sum(np.maximum(0, rs - ys))
    pen_boundary += np.sum(np.maximum(0, ys + rs - h))
    
    # Overlap violations
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    min_dist = rs[:, None] + rs[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    violations = np.maximum(0, min_dist - dist)
    pen_overlap = np.sum(violations[mask])
    
    return obj + penalty * (pen_boundary + pen_overlap)

def _optimize_penalty(x0, w, h, deadline, max_iter=300):
    """
    Run L-BFGS-B with increasing penalty.
    """
    x = x0.copy()
    penalty = 10.0
    n = len(x0) // 3
    
    bounds = []
    for _ in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, min(w, h)/2))
    
    for _ in range(8):
        if time.time() > deadline - 3.0:
            break
        
        def fun(x):
            return _penalty_objective(x, w, h, penalty)
        
        try:
            res = minimize(
                fun, x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x = res.x
        except Exception:
            break
        
        circles = np.array([(x[i*3], x[i*3+1], x[i*3+2]) for i in range(n)])
        if _validate(circles, w, h, tol=1e-4):
            break
        else:
            penalty *= 2.0
    
    return x

def _slsqp_polish(circles, w, h, deadline):
    """
    High-precision polish with SLSQP.
    """
    if time.time() > deadline - 2.0:
        return circles
        
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    def objective(x):
        return -np.sum(x[2::3])
        
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-9})
        
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
            
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-9, min(w, h)/2))
        
    try:
        res = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 150, 'ftol': 1e-14}
        )
        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-8):
                return new_circles
    except Exception:
        pass
        
    return circles

def _final_refine(circles, w, h, deadline):
    """
    Final L-BFGS-B refinement with very tight tolerances.
    """
    if time.time() > deadline - 1.0:
        return circles
    
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-9, min(w, h)/2))
    
    def fun(x):
        return _penalty_objective(x, w, h, 1000.0)
    
    try:
        res = minimize(
            fun, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
        if _validate(new_circles, w, h, tol=1e-8):
            return new_circles
    except Exception:
        pass
        
    return circles

def _solve_aspect(w, h, t0, deadline):
    """
    Solve for a given aspect ratio with multiple initializations.
    """
    best = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = []
    for v in range(4):
        inits.append(_hexagonal_init(w, h, N, variant=v))
    for v in range(2):
        inits.append(_grid_init(w, h, N, variant=v))
    
    for x_init in inits:
        if time.time() > deadline - 5.0:
            break
        
        x_opt = _optimize_penalty(x_init.flatten(), w, h, deadline)
        circles = np.array([(x_opt[i*3], x_opt[i*3+1], x_opt[i*3+2]) for i in range(N)])
        
        if _validate(circles, w, h):
            score = _total_radius(circles)
            if score > best_score:
                best_score = score
                best = circles.copy()
    
    if best is None:
        return None
    
    # Polish
    if time.time() < deadline - 3.0:
        best = _slsqp_polish(best, w, h, deadline)
    
    # Final refine
    if time.time() < deadline - 1.5:
        best = _final_refine(best, w, h, deadline)
    
    return best

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    best_circles = None
    best_score = -np.inf
    best_w = 1.0
    best_h = 1.0
    
    # Perimeter = 2*(w+h) = 4 => w+h=2
    # Try a range of aspect ratios
    ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.67, 0.55, 1.8, 0.5]
    
    for ar in ratios:
        if time.time() > deadline - 2.0:
            break
            
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        orientations = [(w, h)]
        if abs(ar - 1.0) > 1e-3:
            orientations.append((h, w))
            
        for ww, hh in orientations:
            if time.time() > deadline - 2.0:
                break
                
            try:
                circles = _solve_aspect(ww, hh, t0, deadline)
                if circles is not None and len(circles) == N:
                    if _validate(circles, ww, hh):
                        score = _total_radius(circles)
                        if score > best_score:
                            best_score = score
                            best_circles = circles.copy()
                            best_w = ww
                            best_h = hh
            except Exception:
                continue
    
    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final safety check
    if not _validate(best_circles, best_w, best_h):
        for i in range(N):
            best_circles[i, 2] *= 0.999
        if not _validate(best_circles, best_w, best_h):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])
            
    return best_circles
