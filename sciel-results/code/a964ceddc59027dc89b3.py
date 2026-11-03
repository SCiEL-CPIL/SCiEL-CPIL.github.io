"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(circles, w, h, tol=1e-8):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _grid_init(w, h, nc, nr, stagger=False):
    """Generate initial circle positions in a grid pattern."""
    positions = []
    for j in range(nr):
        for i in range(nc):
            if len(positions) >= N:
                break
            x = (i + 0.5) * w / nc
            y = (j + 0.5) * h / nr
            if stagger and j % 2 == 1:
                x += 0.3 * w / nc
            positions.append([x, y])
    # If we don't have enough, add more
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx % 3 - 1) * 0.2 * w
        y = h / 2 + (idx // 3 % 3 - 1) * 0.2 * h
        positions.append([x, y])
    return np.array(positions[:N])

def _local_optimize(w, h, init_params, deadline, maxiter=1000):
    """Local optimization using Nelder-Mead."""
    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        
        # Penalty for out of bounds
        penalty = 0.0
        for i in range(N):
            if pos[i][0] < rad[i]:
                penalty += (rad[i] - pos[i][0]) * 1000
            if pos[i][0] > w - rad[i]:
                penalty += (pos[i][0] - (w - rad[i])) * 1000
            if pos[i][1] < rad[i]:
                penalty += (rad[i] - pos[i][1]) * 1000
            if pos[i][1] > h - rad[i]:
                penalty += (pos[i][1] - (h - rad[i])) * 1000
        
        # Penalty for overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) * 1000
        
        # Maximize sum of radii => minimize negative
        return -np.sum(rad) + penalty
    
    try:
        result = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < 1e9:
            pos = result.x[:2*N].reshape(N, 2)
            rad = result.x[2*N:]
            
            # Verify validity
            circles = np.column_stack([pos, rad])
            if _validate(circles, w, h):
                return result.x, np.sum(rad)
    except Exception:
        pass
    
    return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try multiple aspect ratios
    # Perimeter = 2*(w+h) = 4, so w+h = 2
    # Let's try aspect ratios w/h from 0.5 to 2.0
    aspect_ratios = np.linspace(0.6, 1.8, 12)
    
    # Also try some specific promising ratios
    aspect_ratios = np.unique(np.concatenate([aspect_ratios, [1.0, 1.2, 0.8, 1.5, 0.7]]))
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        
        # w + h = 2, w/h = ar => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try different grid layouts
        layouts = [
            (5, 4, False),   # 5x4 grid
            (4, 5, False),   # 4x5 grid
            (3, 7, False),   # 3x7 grid
            (7, 3, False),   # 7x3 grid
            (6, 4, True),    # 6x4 staggered
            (5, 5, True),    # 5x5 staggered
            (2, 11, False),  # 2x11 grid
            (11, 2, False),  # 11x2 grid
        ]
        
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 5.0:
                break
            
            # Create initial positions
            positions = _grid_init(w, h, nc, nr, stagger)
            
            # Initial radii: estimate based on grid spacing
            cell_w = w / nc
            cell_h = h / nr
            r0 = min(cell_w, cell_h) * 0.4
            radii = np.full(N, r0)
            
            init_params = np.concatenate([positions.flatten(), radii])
            
            # Local optimization
            result_params, value = _local_optimize(w, h, init_params, deadline)
            
            if result_params is not None and value > best_value:
                best_value = value
                pos = result_params[:2*N].reshape(N, 2)
                rad = result_params[2*N:]
                best_result = np.column_stack([pos, rad])
    
    # If no valid result found, return a safe default
    if best_result is None:
        # Fallback: simple grid in a square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
