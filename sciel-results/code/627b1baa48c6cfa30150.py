import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _compute_w_h(aspect):
    """Compute width and height for a rectangle of perimeter 4 with given aspect ratio."""
    w = 2 * aspect / (1 + aspect)
    h = 2 / (1 + aspect)
    return w, h

def _get_initial_positions(w, h, n_circles):
    """Generate diverse initial positions for circles."""
    positions = []
    
    # Grid-based initializations with different densities
    for cols in range(3, 8):
        rows = int(np.ceil(n_circles / cols))
        if cols * rows < n_circles:
            continue
        
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                # Add some jitter to avoid symmetry
                x[idx] = (j + 0.5) * (w / cols) + np.random.uniform(-0.05, 0.05) * w / cols
                y[idx] = (i + 0.5) * (h / rows) + np.random.uniform(-0.05, 0.05) * h / rows
                idx += 1
        
        # Clip to bounds
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        
        positions.append((x, y))
    
    # Random initializations
    for _ in range(5):
        x = np.random.uniform(0, w, n_circles)
        y = np.random.uniform(0, h, n_circles)
        positions.append((x, y))
    
    # Hexagonal-like packing
    x = np.zeros(n_circles)
    y = np.zeros(n_circles)
    idx = 0
    row = 0
    while idx < n_circles:
        for col in range(10):
            if idx >= n_circles:
                break
            offset = (col % 2) * 0.5
            x[idx] = (col + 0.5 + offset) * w / 10
            y[idx] = (row + 0.5) * h / 8
            idx += 1
        row += 1
    x = np.clip(x, 0, w)
    y = np.clip(y, 0, h)
    positions.append((x, y))
    
    return positions

def _force_relax(x, y, r, w, h, max_iter=2000, tol=1e-8):
    """Force-directed relaxation to resolve overlaps and maximize radii."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    n = len(r)
    dr = 0.002
    
    for iteration in range(max_iter):
        # Try to grow all radii
        r_new = r + dr
        
        # Check if growth is feasible (no overlaps, within bounds)
        feasible = True
        
        # Check bounds
        if np.any(x - r_new < -1e-6) or np.any(x + r_new > w + 1e-6) or \
           np.any(y - r_new < -1e-6) or np.any(y + r_new > h + 1e-6):
            # Can't grow uniformly, try smaller growth
            dr = dr * 0.5
            if dr < 1e-10:
                break
            continue
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r_new[i] + r_new[j] - 1e-8:
                    feasible = False
                    break
            if not feasible:
                break
        
        if not feasible:
            # Resolve overlaps
            for _ in range(10):  # Inner iterations
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        dist = np.sqrt(dx * dx + dy * dy)
                        min_dist = r[i] + r[j]
                        
                        if dist < min_dist:
                            if dist < 1e-8:
                                dx = 0.01
                                dy = 0.0
                                dist = 0.01
                            
                            overlap = min_dist - dist
                            push = overlap * 0.5
                            x[i] -= (dx / dist) * push
                            y[i] -= (dy / dist) * push
                            x[j] += (dx / dist) * push
                            y[j] += (dy / dist) * push
                            
                            # Shrink radii slightly
                            shrink = overlap * 0.05
                            r[i] = max(r[i] - shrink, 1e-4)
                            r[j] = max(r[j] - shrink, 1e-4)
                
                # Enforce bounds
                for i in range(n):
                    x[i] = np.clip(x[i], r[i], w - r[i])
                    y[i] = np.clip(y[i], r[i], h - r[i])
            
            # Try smaller growth
            dr = max(dr * 0.7, 1e-10)
            if dr < 1e-10:
                break
        else:
            # Growth is feasible, accept it
            r = r_new
            # Reset growth step for next iteration
            dr = min(dr * 1.1, 0.01)
    
    return x, y, r

def _polish_with_slsqp(x, y, r, w, h, max_iter=100):
    """Use SLSQP to polish the solution."""
    n = len(r)
    
    # Variables: x, y, r
    def objective(params):
        x_p = params[:n]
        y_p = params[n:2*n]
        r_p = params[2*n:]
        return -np.sum(r_p)
    
    # Constraints: no overlaps and within bounds
    def constraints():
        cons = []
        
        # Boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})  # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (w - p[i]) - p[2*n+i]})  # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})  # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (h - p[n+i]) - p[2*n+i]})  # h - y_i - r_i >= 0
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (p[j] - p[i])**2 + (p[n+j] - p[n+i])**2 - (p[2*n+i] + p[2*n+j])**2
                })
        
        # Positive radius
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i] - 1e-6})
        
        return cons
    
    x0 = np.concatenate([x, y, r])
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints(),
            bounds=[(0, w)] * n + [(0, h)] * n + [(1e-6, np.max([w, h]) / 2)] * n,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        
        if result.success:
            x_new = result.x[:n]
            y_new = result.x[n:2*n]
            r_new = result.x[2*n:]
            return x_new, y_new, r_new
    except:
        pass
    
    return x, y, r

def _is_valid(x, y, r, w, h, tol=1e-6):
    """Check if the configuration is valid."""
    n = len(r)
    
    # Check bounds
    if np.any(x - r < -tol) or np.any(x + r > w + tol) or \
       np.any(y - r < -tol) or np.any(y + r > h + tol):
        return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_sum = -np.inf
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = np.linspace(0.7, 1.4, 15)
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        w, h = _compute_w_h(aspect)
        
        # Get initial positions
        initial_positions = _get_initial_positions(w, h, N)
        
        for x_init, y_init in initial_positions:
            if time.time() > deadline:
                break
            
            # Initial radius estimate
            r_init = np.sqrt(w * h / (N * np.pi * 0.9))
            r_init = np.clip(r_init, 0.05, 0.2)
            r = np.full(N, r_init)
            
            # Force relaxation
            x_cur, y_cur, r_cur = _force_relax(x_init, y_init, r, w, h, max_iter=1000)
            
            # Polish with SLSQP
            x_pol, y_pol, r_pol = _polish_with_slsqp(x_cur, y_cur, r_cur, w, h, max_iter=50)
            
            # Validate
            if _is_valid(x_pol, y_pol, r_pol, w, h):
                sum_r = np.sum(r_pol)
                if sum_r > best_sum:
                    best_sum = sum_r
                    best_result = np.column_stack((x_pol, y_pol, r_pol))
        
        # If we have a good result, break early
        if best_sum > 1.5:
            break
    
    # If no valid result found, return a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        w, h = _compute_w_h(1.0)
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * (w / side)
                y[idx] = (i + 0.5) * (h / side)
                r[idx] = min(w / side, h / side) * 0.45
                idx += 1
        
        best_result = np.column_stack((x, y, r))
    
    return best_result
