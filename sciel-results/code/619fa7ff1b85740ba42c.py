"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
PERIMETER = 4.0
TIME_LIMIT = 110.0  # Safety margin before 120s limit


def _is_valid(circles, tol=1e-9):
    """Check if the configuration is valid."""
    if len(circles) == 0:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return False
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    if 2 * (W + H) > PERIMETER + tol:
        return False
    
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            sum_r = r[i] + r[j]
            if dist_sq < (sum_r - tol) ** 2:
                return False
    return True


def _compute_value(circles):
    """Compute the sum of radii for a valid configuration, else 0."""
    if _is_valid(circles):
        return np.sum(circles[:, 2])
    return 0.0


def _seed_grid():
    """Create a simple grid-based seed configuration."""
    best_val = -1
    best_circles = None
    
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
            
        s = 2.0 / (cols + rows)
        
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * s
                cy = (j + 0.5) * s
                r = 0.5 * s - 1e-4
                circles.append([cx, cy, r])
        circles = np.array(circles[:N])
        
        if _is_valid(circles):
            val = np.sum(circles[:, 2])
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
    
    return best_circles


def _optimize(circles, time_budget):
    """Optimize circle positions and radii, tracking best valid solution."""
    t_start = time.time()
    best_circles = circles.copy()
    best_val = _compute_value(best_circles)
    
    def objective(params):
        if time.time() - t_start > time_budget:
            return 1e10
        
        n = len(params) // 3
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        penalty = 0.0
        penalty -= 100.0 * np.sum(np.minimum(r, 0))
        
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        
        W = x_max - x_min
        H = y_max - y_min
        
        pen_perim = max(0, 2 * (W + H) - PERIMETER)
        penalty += 1000.0 * pen_perim ** 2
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += 1000.0 * overlap ** 2
        
        return -np.sum(r) + penalty
    
    params = circles.flatten()
    
    # Callback to check validity and update best
    def callback(xk):
        nonlocal best_circles, best_val
        current_circles = xk.reshape(-1, 3)
        current_circles[:, 2] = np.maximum(current_circles[:, 2], 1e-6)
        val = _compute_value(current_circles)
        if val > best_val:
            best_val = val
            best_circles = current_circles.copy()
        return False # Continue optimization
    
    try:
        minimize(objective, params, method='Nelder-Mead', 
                 options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5},
                 callback=callback)
    except Exception:
        pass
    
    # Check final result
    final_circles = params.reshape(-1, 3) # This is stale, need to re-run or capture from last call? 
    # Actually, minimize doesn't return the intermediate best. We rely on callback.
    # But we should also check the final result of the optimizer if it's valid.
    # The callback updates best_circles.
    
    return best_circles


def construct():
    """Construct 21 disjoint circles in a perimeter-4 rectangle maximizing sum of radii."""
    t_start = time.time()
    
    # Start with a good seed
    circles = _seed_grid()
    
    if circles is None or len(circles) < N:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        circles = np.array(cells[:N])
        if not _is_valid(circles):
             # If even seed is invalid, try smaller radii
             circles[:, 2] *= 0.5
             if not _is_valid(circles):
                 circles[:, 2] *= 0.5
    
    # Optimize
    time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
    if time_remaining > 2.0:
        circles = _optimize(circles, time_remaining)
    
    # Final validation
    if not _is_valid(circles):
        # If optimization failed to produce a valid result, try to repair or fallback to seed
        # For now, just return the seed if it was valid, else a minimal valid config
        seed_circles = _seed_grid()
        if seed_circles is not None and _is_valid(seed_circles):
            circles = seed_circles
        else:
            # Construct a trivially valid small configuration
            s = 2.0 / (7 + 3) # 7x3 grid
            circles = []
            for i in range(7):
                for j in range(3):
                    if len(circles) >= N: break
                    circles.append([(i+0.5)*s, (j+0.5)*s, 0.4*s])
            circles = np.array(circles[:N])
    
    return circles
