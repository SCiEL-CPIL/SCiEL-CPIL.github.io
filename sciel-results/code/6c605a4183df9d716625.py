"""
Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Strategy:
1. Use a vectorized penalty-based L-BFGS-B optimizer for speed.
2. Use hexagonal grid initialization for better starting topology.
3. Explore a wide range of aspect ratios.
4. Apply a final SLSQP polish step on the best candidate for high precision.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # Safety margin under 150s wall-clock

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

def _hexagonal_init(w, h, n):
    """
    Generate a hexagonal packing initialization.
    This is generally a better starting point for circle packing than a grid.
    """
    circles = []
    # Estimate side length of hexagon cell
    # Area per circle ~ (2*r)^2 * sqrt(3)/2 => r ~ sqrt(2/sqrt(3) * A/n)
    # For a square, r ~ 0.5/sqrt(n) * sqrt(2/sqrt(3)) ~ 0.5/sqrt(n) * 1.07
    # Let's just try to fit rows and columns based on hex geometry.
    
    # Hexagonal lattice:
    # dx = 2*r
    # dy = r * sqrt(3)
    # We want to maximize r.
    # Let's assume a grid of R rows and C cols.
    # Width = C * 2*r + (C-1)*r (if staggered, width is roughly C*1.5*r + 0.5*r? No, simple hex grid width is C*2r if aligned, but staggered rows shift by r.
    # Let's use a simple grid first but with hexagonal spacing if possible, or just a dense grid.
    # Actually, for 21 circles, a 4x5 or 5x4 grid is common.
    # Hexagonal packing of 21:
    # Row 1: 4 circles
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles? No, 4+5+4+5+3 = 21.
    
    # Let's just use a rectangular grid initialization which is robust, 
    # but optimized for aspect ratio.
    # The prompt says "hexagonal grid initialization starts closer to optimal".
    # Let's implement a simple hexagonal layout for 21 circles.
    
    # Layout:
    # 4 5 4 5 3
    # Or 5 4 5 4 3
    # Let's try to generate centers.
    
    # Simpler approach: Just use a grid that fills the space well.
    # For 21, 4x5 is 20, so one extra.
    # Let's stick to a grid but with a slight hexagonal shift if possible?
    # No, let's just use a standard grid but with better radius estimation.
    
    # Actually, let's use the specific hexagonal arrangement for 21.
    # Rows: 4, 5, 4, 5, 3
    rows = [4, 5, 4, 5, 3]
    total_circles = sum(rows)
    if total_circles < n:
        # Add more rows or circles
        while sum(rows) < n:
            rows.append(4)
        # Trim if necessary
        while sum(rows) > n:
            rows[-1] -= 1
            if rows[-1] <= 0:
                rows.pop()
    
    # Calculate optimal r for this layout in a unit square (or w,h)
    # Width of row k with c circles: c * 2r + (c-1)*r? No.
    # In hex packing, horizontal spacing is 2r. Vertical spacing is r*sqrt(3).
    # Row i has c_i circles.
    # The horizontal extent is max(c_i) * 2r.
    # The vertical extent is (len(rows) - 1) * r * sqrt(3) + 2r.
    
    max_circles = max(rows)
    num_rows = len(rows)
    
    # We want to fit this in w x h.
    # w >= max_circles * 2r  => r <= w / (2 * max_circles)
    # h >= (num_rows - 1) * r * sqrt(3) + 2r => r <= h / ((num_rows - 1) * sqrt(3) + 2)
    
    r_w = w / (2.0 * max_circles)
    r_h = h / ((num_rows - 1) * np.sqrt(3) + 2.0)
    r = min(r_w, r_h)
    
    # Center the layout
    total_width = max_circles * 2 * r
    total_height = (num_rows - 1) * r * np.sqrt(3) + 2 * r
    
    offset_x = (w - total_width) / 2.0
    offset_y = (h - total_height) / 2.0
    
    # Generate positions
    y = offset_y + r
    for i, count in enumerate(rows):
        x_start = offset_x + r
        # Stagger every other row
        if i % 2 == 1:
            x_start += r
            
        for j in range(count):
            cx = x_start + j * 2 * r
            cy = y
            # Clamp to bounds just in case
            if cx + r > w: cx = w - r
            if cx - r < 0: cx = r
            if cy + r > h: cy = h - r
            if cy - r < 0: cy = r
            circles.append([cx, cy, r])
            
        y += r * np.sqrt(3)
        
    # If we have fewer circles than N, add more?
    # The logic above should handle N=21 exactly with the specific row counts.
    # If N is different, we might have issues. Let's fallback to grid if mismatch.
    if len(circles) != n:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = 0.5 * min(w, h) / side * 0.9
                cells.append([cx, cy, r])
        return np.array(cells[:n])
        
    return np.array(circles[:n])

def _penalty_objective(x, w, h, penalty):
    """
    Vectorized penalty objective function.
    x: array of shape (3*n,) [x0, y0, r0, x1, y1, r1, ...]
    """
    n = len(x) // 3
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Objective: maximize sum of radii => minimize -sum(rs)
    obj = -np.sum(rs)
    
    # Penalties
    # 1. Boundary violations
    # x - r >= 0, x + r <= w, y - r >= 0, y + r <= h
    # Violation = max(0, r - x) + max(0, x + r - w) + ...
    pen_boundary = 0.0
    pen_boundary += np.sum(np.maximum(0, rs - xs))
    pen_boundary += np.sum(np.maximum(0, xs + rs - w))
    pen_boundary += np.sum(np.maximum(0, rs - ys))
    pen_boundary += np.sum(np.maximum(0, ys + rs - h))
    
    # 2. Overlap violations
    # dist(i,j) >= ri + rj
    # Vectorized distance matrix
    # Use broadcasting
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    min_dist = rs[:, None] + rs[None, :]
    
    # Only upper triangle
    # Create mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    violations = np.maximum(0, min_dist - dist)
    pen_overlap = np.sum(violations[mask])
    
    return obj + penalty * (pen_boundary + pen_overlap)

def _optimize_penalty(x0, w, h, deadline, max_iter=500, penalty_start=10.0, penalty_growth=2.0):
    """
    Run L-BFGS-B with increasing penalty.
    """
    x = x0.copy()
    penalty = penalty_start
    
    for i in range(10): # Max 10 penalty stages
        if time.time() > deadline - 5.0:
            break
            
        # Define objective for this penalty stage
        def fun(x):
            return _penalty_objective(x, w, h, penalty)
            
        # Bounds
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        bounds = []
        for _ in range(len(x0)//3):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0, min(w, h)/2))
            
        try:
            res = minimize(
                fun, x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x = res.x
        except Exception:
            break
            
        # Check validity to decide if we need to increase penalty
        # Convert x to circles
        circles = np.array([(x[i*3], x[i*3+1], x[i*3+2]) for i in range(len(x0)//3)])
        if _validate(circles, w, h, tol=1e-4):
            # If valid, we might be done, or we can try to reduce penalty to allow larger radii?
            # Actually, if valid, the penalty term is 0. The objective is just -sum(r).
            # L-BFGS-B will push r as large as possible subject to the soft constraints becoming hard enough.
            # If it's valid, we can stop increasing penalty and just return.
            # However, we might want to run one more iteration with lower penalty to ensure we aren't stuck on the boundary of the penalty region?
            # No, if valid, the gradient of the penalty is 0. The gradient of the objective is -1 for radii.
            # So it will increase radii until it hits a constraint.
            # If it hits a constraint and stays valid, it's a local optimum for this topology.
            break
        else:
            penalty *= penalty_growth
            
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
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-8})
        
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
        bounds.append((1e-8, min(w, h)/2))
        
    try:
        res = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-8):
                return new_circles
    except Exception:
        pass
        
    return circles

def _solve_aspect(w, h, t0, deadline):
    # Initialize with hexagonal grid
    x_init = _hexagonal_init(w, h, N)
    
    # Flatten for optimizer
    x_flat = x_init.flatten()
    
    # Optimize with penalty method
    x_opt = _optimize_penalty(x_flat, w, h, deadline)
    
    # Reshape
    circles = np.array([(x_opt[i*3], x_opt[i*3+1], x_opt[i*3+2]) for i in range(N)])
    
    # Polish with SLSQP if time permits
    if time.time() < deadline - 5.0:
        circles = _slsqp_polish(circles, w, h, deadline)
        
    return circles

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    best_circles = None
    best_score = -np.inf
    best_w = 1.0
    best_h = 1.0
    
    # Perimeter = 2*(w+h) = 4 => w+h=2.
    # Aspect ratio ar = w/h.
    # w = ar*h => h(ar+1)=2 => h = 2/(ar+1), w = 2*ar/(ar+1).
    
    # Try a range of aspect ratios
    # Symmetric around 1.0
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.5]
    
    for ar in ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Also try swapped orientation if ar != 1
        orientations = [(w, h)]
        if abs(ar - 1.0) > 1e-3:
            orientations.append((h, w))
            
        for ww, hh in orientations:
            if time.time() > deadline - 1.0:
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
