import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _objective_and_grad(x, w, h, n_circles, penalty_factor):
    """
    Objective: minimize -sum(r) + penalty
    x = [x1, y1, r1, ..., xn, yn, rn]
    """
    x_coords = x[0:n_circles*2:2]
    y_coords = x[1:n_circles*2:2]
    r_coords = x[2:n_circles*2:2]
    
    val = -np.sum(r_coords)
    
    grad = np.zeros_like(x)
    grad[2:n_circles*2:2] = -1.0
    
    # Circle-circle overlap penalty
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = x_coords[i] - x_coords[j]
            dy = y_coords[i] - y_coords[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-8
            
            if dist < 1e-8:
                dist = 1e-8
                dx = 1e-8
                dy = 0.0
            
            sum_r = r_coords[i] + r_coords[j]
            violation = sum_r - dist
            
            if violation > 0:
                val += penalty_factor * violation**2
                g_penalty = 2.0 * penalty_factor * violation
                grad[2*i] += -g_penalty * (dx / dist)
                grad[2*i+1] += -g_penalty * (dy / dist)
                grad[2*i+2] += g_penalty
                grad[2*j] += g_penalty * (dx / dist)
                grad[2*j+1] += g_penalty * (dy / dist)
                grad[2*j+2] += g_penalty
    
    # Wall collision penalties
    for i in range(n_circles):
        r = r_coords[i]
        x = x_coords[i]
        y = y_coords[i]
        
        # Left wall
        v = r - x
        if v > 0:
            val += penalty_factor * v**2
            grad[2*i] += 2.0 * penalty_factor * v * (-1.0)
            grad[2*i+2] += 2.0 * penalty_factor * v * 1.0
            
        # Right wall
        v = r - (w - x)
        if v > 0:
            val += penalty_factor * v**2
            grad[2*i] += 2.0 * penalty_factor * v * (1.0)
            grad[2*i+2] += 2.0 * penalty_factor * v * 1.0
            
        # Bottom wall
        v = r - y
        if v > 0:
            val += penalty_factor * v**2
            grad[2*i+1] += 2.0 * penalty_factor * v * (-1.0)
            grad[2*i+2] += 2.0 * penalty_factor * v * 1.0
            
        # Top wall
        v = r - (h - y)
        if v > 0:
            val += penalty_factor * v**2
            grad[2*i+1] += 2.0 * penalty_factor * v * (1.0)
            grad[2*i+2] += 2.0 * penalty_factor * v * 1.0
            
    return val, grad

def _validate_solution(x, w, h, n_circles, tol=1e-5):
    """Check if solution satisfies all constraints within tolerance."""
    x_coords = x[0:n_circles*2:2]
    y_coords = x[1:n_circles*2:2]
    r_coords = x[2:n_circles*2:2]
    
    for i in range(n_circles):
        xi = x_coords[i]
        yi = y_coords[i]
        ri = r_coords[i]
        
        if ri < -tol:
            return False
        if xi < -tol or xi > w + tol:
            return False
        if yi < -tol or yi > h + tol:
            return False
        if ri - xi < -tol or ri - (w - xi) < -tol:
            return False
        if ri - yi < -tol or ri - (h - yi) < -tol:
            return False
    
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dist = np.sqrt((x_coords[i]-x_coords[j])**2 + (y_coords[i]-y_coords[j])**2)
            if dist < (r_coords[i] + r_coords[j]) - tol:
                return False
    
    return True

def _fix_solution(x, w, h, n_circles):
    """Clip solution to be strictly valid."""
    x_fixed = x.copy()
    for i in range(n_circles):
        x_fixed[2*i] = np.clip(x_fixed[2*i], 0, w)
        x_fixed[2*i+1] = np.clip(x_fixed[2*i+1], 0, h)
        x_fixed[2*i+2] = max(0, x_fixed[2*i+2])
        # Ensure circle fits in walls
        x_fixed[2*i+2] = min(x_fixed[2*i+2], x_fixed[2*i], w - x_fixed[2*i], x_fixed[2*i+1], h - x_fixed[2*i+1])
    
    # Resolve overlaps by shrinking radii
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dist = np.sqrt((x_fixed[2*i]-x_fixed[2*j])**2 + (x_fixed[2*i+1]-x_fixed[2*j+1])**2)
            max_allowed = max(0, (dist - 1e-6) / 2.0)
            if x_fixed[2*i+2] > max_allowed:
                x_fixed[2*i+2] = max_allowed
            if x_fixed[2*j+2] > max_allowed:
                x_fixed[2*j+2] = max_allowed
            # Re-check wall constraints after shrinking
            x_fixed[2*i+2] = min(x_fixed[2*i+2], x_fixed[2*i], w - x_fixed[2*i], x_fixed[2*i+1], h - x_fixed[2*i+1])
            x_fixed[2*j+2] = min(x_fixed[2*j+2], x_fixed[2*j], w - x_fixed[2*j], x_fixed[2*j+1], h - x_fixed[2*j+1])
    
    return x_fixed

def _pack_for_aspect(w, h, num_starts=4, time_limit=30, deadline=None):
    """Pack circles in w x h rectangle using L-BFGS-B with penalty method."""
    n_circles = N
    n_vars = n_circles * 3
    max_r = min(w, h) / 2.0
    
    bounds = []
    for i in range(n_circles):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
    
    best_val = np.inf
    best_x = None
    
    start_time = time.time()
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        if deadline is not None and time.time() > deadline:
            break
        
        # Grid initialization with noise
        cols = int(np.ceil(np.sqrt(n_circles * w / h)))
        rows = int(np.ceil(n_circles / cols))
        
        x0 = np.zeros(n_vars)
        for i in range(n_circles):
            col = i % cols
            row = i // cols
            cx = (col + 0.5) * w / cols
            cy = (row + 0.5) * h / rows
            noise = 0.15 * min(w/cols, h/rows)
            x0[2*i] = np.clip(cx + np.random.uniform(-noise, noise), 0.01, w-0.01)
            x0[2*i+1] = np.clip(cy + np.random.uniform(-noise, noise), 0.01, h-0.01)
            x0[2*i+2] = 0.4 * min(w/cols, h/rows)
        
        # Multi-stage penalty optimization
        penalty_schedule = [50.0, 200.0, 1000.0, 5000.0]
        current_x = x0
        
        for stage, penalty in enumerate(penalty_schedule):
            if time.time() - start_time > time_limit:
                break
            if deadline is not None and time.time() > deadline:
                break
            
            def callback(Xk):
                return (time.time() - start_time > time_limit or 
                       (deadline is not None and time.time() > deadline))
            
            try:
                res = minimize(
                    lambda x: _objective_and_grad(x, w, h, n_circles, penalty)[0],
                    current_x,
                    jac=lambda x: _objective_and_grad(x, w, h, n_circles, penalty)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'maxfun': 5000, 'callback': callback}
                )
                current_x = res.x
            except Exception:
                pass
        
        # Validate and update best
        if _validate_solution(current_x, w, h, n_circles):
            val = -np.sum(current_x[2:n_circles*2:2])
            if val < best_val:
                best_val = val
                best_x = current_x.copy()
        else:
            # Try to fix
            fixed_x = _fix_solution(current_x, w, h, n_circles)
            if _validate_solution(fixed_x, w, h, n_circles):
                val = -np.sum(fixed_x[2:n_circles*2:2])
                if val < best_val:
                    best_val = val
                    best_x = fixed_x.copy()
    
    if best_x is None:
        return None, 0.0
    
    positions = np.column_stack([best_x[0::2], best_x[1::2]])
    radii = np.maximum(best_x[2::2], 0)
    score = np.sum(radii)
    
    return np.column_stack([positions, radii]), score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios: w + h = 2 (perimeter 4)
    aspect_configs = [
        (1.0, 1.0),      # Square
        (1.1, 0.9),
        (0.9, 1.1),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.5, 0.5),
        (0.5, 1.5),
    ]
    
    for w, h in aspect_configs:
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 15:
            break
        
        # Allocate time: more for promising aspect ratios
        num_starts = 4 if time_remaining > 80 else 2
        time_per_aspect = min(time_remaining - 5, 30)
        
        result, score = _pack_for_aspect(w, h, num_starts=num_starts, 
                                         time_limit=time_per_aspect, deadline=deadline)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result
    
    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        cols, rows = 5, 5
        positions = []
        radii = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.95
                positions.append(pos)
                radii.append(r)
                count += 1
        best_result = np.column_stack([np.array(positions), np.array(radii)])
    
    return best_result
