"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 110  # safety margin before 120s limit


def _perimeter_constraint(w, h):
    """Return perimeter = 2*(w+h). We want this <= 4."""
    return 2.0 * (w + h)


def _check_validity(circles, w, h):
    """Check if circles are valid: within [0,w]x[0,h], disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -1e-9 or x + r > w + 1e-9:
            return False
        if y - r < -1e-9 or y + r > h + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            if dist_sq < min_dist ** 2 - 1e-9:
                return False
    return True


def _objective(params, w, h):
    """
    Minimize negative sum of radii plus penalty for constraint violations.
    params: [x1,y1,r1, x2,y2,r2, ..., w, h]
    """
    n = N
    circles = params[:3 * n].reshape(n, 3)
    w_val = params[3 * n]
    h_val = params[3 * n + 1]
    
    if w_val <= 0 or h_val <= 0:
        return 1e10
    
    # Perimeter constraint
    perim = 2.0 * (w_val + h_val)
    perim_violation = max(0, perim - 4.0)
    
    # Boundary constraints
    penalty = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            penalty += 1e6 * (-r) ** 2
        # Boundary: x in [r, w-r], y in [r, h-r]
        if x < r:
            penalty += 1e4 * (r - x) ** 2
        if x > w_val - r:
            penalty += 1e4 * (x - (w_val - r)) ** 2
        if y < r:
            penalty += 1e4 * (r - y) ** 2
        if y > h_val - r:
            penalty += 1e4 * (y - (h_val - r)) ** 2
    
    # Disjointness
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            min_dist = ri + rj
            if dist < min_dist:
                penalty += 1e4 * (min_dist - dist) ** 2
    
    total_penalty = 1e6 * perim_violation ** 2 + penalty
    sum_radii = np.sum(circles[:, 2])
    
    return total_penalty - sum_radii


def _make_initial_grid():
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((x, y, r))
    return cells


def _optimize_from_init(circles_init, w_init, h_init, time_budget=20.0):
    """
    Optimize from an initial configuration using SLSQP.
    Returns (best_value, best_circles, best_w, best_h) or None.
    """
    n = N
    x0 = np.zeros(3 * n + 2)
    for i in range(n):
        x0[3 * i] = circles_init[i][0]
        x0[3 * i + 1] = circles_init[i][1]
        x0[3 * i + 2] = circles_init[i][2]
    x0[3 * n] = w_init
    x0[3 * n + 1] = h_init
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 2))  # x
        bounds.append((0, 2))  # y
        bounds.append((0, 1))  # r
    bounds.append((0.1, 2))  # w
    bounds.append((0.1, 2))  # h
    
    start_time = time.time()
    
    try:
        result = minimize(
            _objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
        
        elapsed = time.time() - start_time
        if result.success or result.fun < 1e6:
            best_params = result.x
            circles = best_params[:3 * n].reshape(n, 3)
            w_val = best_params[3 * n]
            h_val = best_params[3 * n + 1]
            
            # Check validity
            if _check_validity(circles, w_val, h_val):
                value = np.sum(circles[:, 2])
                return value, circles, w_val, h_val
        return None
    except Exception:
        return None


def _try_aspect_ratio(w, h, time_budget=15.0):
    """
    Try to optimize with a fixed aspect ratio w/h = ratio.
    """
    n = N
    # Parameterize: use (x1,y1,r1,...,xn,yn,rn, w) where h = 4/2 - w = 2 - w
    # Actually perimeter = 2*(w+h) = 4 => w+h = 2 => h = 2 - w
    # So we only need to optimize w in (0, 2)
    
    def obj_fixed_perim(params):
        circles = params[:3 * n].reshape(n, 3)
        w_val = params[3 * n]
        h_val = 2.0 - w_val
        
        if w_val <= 0.01 or h_val <= 0.01:
            return 1e10
        
        penalty = 0.0
        for i in range(n):
            x, y, r = circles[i]
            if r < 0:
                penalty += 1e6 * (-r) ** 2
            if x < r:
                penalty += 1e4 * (r - x) ** 2
            if x > w_val - r:
                penalty += 1e4 * (x - (w_val - r)) ** 2
            if y < r:
                penalty += 1e4 * (r - y) ** 2
            if y > h_val - r:
                penalty += 1e4 * (y - (h_val - r)) ** 2
        
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                min_dist = ri + rj
                if dist < min_dist:
                    penalty += 1e4 * (min_dist - dist) ** 2
        
        sum_radii = np.sum(circles[:, 2])
        return 1e4 * penalty - sum_radii
    
    # Initial guess: square-ish
    w0 = 1.0
    circles_init = _make_initial_grid()
    x0 = np.zeros(3 * n + 1)
    for i in range(n):
        x0[3 * i] = circles_init[i][0] * w0
        x0[3 * i + 1] = circles_init[i][1] * (2.0 - w0)
        x0[3 * i + 2] = circles_init[i][2] * min(w0, 2.0 - w0)
    x0[3 * n] = w0
    
    bounds = []
    for i in range(n):
        bounds.append((0, 2))
        bounds.append((0, 2))
        bounds.append((0, 1))
    bounds.append((0.05, 1.95))
    
    start_time = time.time()
    
    try:
        result = minimize(
            obj_fixed_perim,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.fun < 1e6:
            best_params = result.x
            circles = best_params[:3 * n].reshape(n, 3)
            w_val = best_params[3 * n]
            h_val = 2.0 - w_val
            
            if _check_validity(circles, w_val, h_val):
                value = np.sum(circles[:, 2])
                return value, circles, w_val, h_val
        return None
    except Exception:
        return None


def construct():
    start_time = time.time()
    best_value = 0.0
    best_result = None
    
    # Strategy 1: Grid initialization with full SLSQP optimization
    circles_init = _make_initial_grid()
    w_init = 1.0
    h_init = 1.0
    
    result1 = _optimize_from_init(circles_init, w_init, h_init, time_budget=30.0)
    if result1 is not None:
        val, circ, w, h = result1
        if val > best_value:
            best_value = val
            best_result = np.column_stack([circ, np.ones(N) * 0])  # placeholder
            # Store properly
            best_circles = circ.copy()
            best_w = w
            best_h = h
    
    # Strategy 2: Fixed perimeter optimization (w+h=2)
    result2 = _try_aspect_ratio(1.0, 1.0, time_budget=25.0)
    if result2 is not None:
        val, circ, w, h = result2
        if val > best_value:
            best_value = val
            best_circles = circ.copy()
            best_w = w
            best_h = h
    
    # Strategy 3: Different aspect ratios
    for aspect in [0.7, 1.4, 0.8, 1.25]:
        if time.time() - start_time > DEADLINE - 20:
            break
        w_try = aspect / (1 + aspect) * 2.0
        h_try = 2.0 - w_try
        if w_try <= 0.05 or h_try <= 0.05:
            continue
        
        # Scale initial grid to fit
        circles_scaled = []
        for x, y, r in circles_init:
            circles_scaled.append((x * w_try, y * h_try, r * min(w_try, h_try)))
        
        result3 = _optimize_from_init(circles_scaled, w_try, h_try, time_budget=15.0)
        if result3 is not None:
            val, circ, w, h = result3
            if val > best_value:
                best_value = val
                best_circles = circ.copy()
                best_w = w
                best_h = h
    
    # Final check
    if best_result is not None and best_circles is not None:
        # Format: (x, y, r) for each circle
        # The return should be array of shape (n, 3)
        # We need to verify the bounding rectangle has perimeter <= 4
        # The circles are already in absolute coordinates
        
        # Verify once more
        if _check_validity(best_circles, best_w, best_h):
            # Ensure perimeter is exactly 4 (or slightly less)
            perim = 2.0 * (best_w + best_h)
            if perim > 4.0 + 1e-6:
                # Scale down
                scale = 4.0 / perim
                best_circles[:, 0] *= scale
                best_circles[:, 1] *= scale
                best_circles[:, 2] *= scale
                best_w *= scale
                best_h *= scale
            
            return best_circles
    
    # Fallback: return the initial grid scaled to fit perimeter 4
    circles_init = _make_initial_grid()
    # Scale to fit in [0,1]x[0,1] which has perimeter 4
    result = np.array(circles_init[:N])
    return result
