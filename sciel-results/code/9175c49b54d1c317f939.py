"""Circle packing in perimeter-4 rectangle using force-directed relaxation with multi-aspect search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    best_value = 0.0
    best_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 0.9, 1.1, 0.75, 1.33]
    
    for aspect_idx, aspect in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
        
        # For perimeter 4: w + h = 2, so h = 2 - w
        # aspect = w/h => w = aspect * h => aspect*h + h = 2 => h = 2/(aspect+1)
        h_rect = 2.0 / (aspect + 1.0)
        w_rect = aspect * h_rect
        
        # Try multiple random initializations for this aspect
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(attempt * 1000 + aspect_idx * 100 + 42)
            
            # Generate initial positions within the rectangle
            centers = np.column_stack([
                rng.uniform(0.1, w_rect - 0.1, N),
                rng.uniform(0.1, h_rect - 0.1, N)
            ])
            radii = np.full(N, 0.07)
            
            # Optimize using force-directed approach
            result = _force_directed_optimize(w_rect, h_rect, centers.copy(), radii.copy(), 
                                              deadline - (time.time() - start_time))
            if result is not None:
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result
            
            if time.time() > deadline:
                break
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _force_directed_optimize(w_rect, h_rect, centers, radii, time_budget):
    """Optimize circle positions and radii using force-directed relaxation."""
    start_time = time.time()
    n = len(centers)
    
    # Parameter vector: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    def objective(params):
        centers_opt = params[:2*n].reshape(n, 2)
        radii_opt = params[2*n:3*n]
        
        # Penalty for circles outside rectangle
        penalty = 0.0
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            if x - r < 0:
                penalty += (x - r) ** 2 * 1000
            if x + r > w_rect:
                penalty += (x + r - w_rect) ** 2 * 1000
            if y - r < 0:
                penalty += (y - r) ** 2 * 1000
            if y + r > h_rect:
                penalty += (y + r - h_rect) ** 2 * 1000
        
        # Penalty for overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii_opt[i] + radii_opt[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 500
        
        # Maximize sum of radii (negate for minimization)
        return -np.sum(radii_opt) + penalty
    
    # Initial parameters
    params_init = np.concatenate([centers.flatten(), radii])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.01, w_rect - 0.01))  # x
        bounds.append((0.01, h_rect - 0.01))  # y
        bounds.append((0.02, min(w_rect, h_rect) / 2 - 0.01))  # r
    
    # Optimize
    try:
        res = minimize(objective, params_init, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
        
        if res.success or res.fun < 1e4:
            centers_opt = res.x[:2*n].reshape(n, 2)
            radii_opt = res.x[2*n:3*n]
            
            # Verify solution is valid
            valid = True
            for i in range(n):
                x, y = centers_opt[i]
                r = radii_opt[i]
                if x - r < -1e-6 or x + r > w_rect + 1e-6 or y - r < -1e-6 or y + r > h_rect + 1e-6:
                    valid = False
                    break
                if r < 0:
                    valid = False
                    break
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = centers_opt[i, 0] - centers_opt[j, 0]
                        dy = centers_opt[i, 1] - centers_opt[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < radii_opt[i] + radii_opt[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                result = np.column_stack([centers_opt, radii_opt])
                return result
    except Exception:
        pass
    
    return None
