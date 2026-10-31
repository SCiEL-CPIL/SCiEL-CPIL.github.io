import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    best_score = 0.0
    best_result = None
    
    # Try various aspect ratios
    # w + h = 2
    # ar = w/h => w = 2*ar/(1+ar), h = 2/(1+ar)
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.7, 2.5, 0.5, 1.1, 1.4, 1.8]
    
    # Number of random restarts for greedy placement
    n_restarts = 5
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 1000 + int(ar * 10))
            
            # Greedy Placement
            circles = [] # list of (x, y, r)
            
            for k in range(N):
                best_x, best_y, best_r = None, None, 0.0
                
                # Generate candidate points
                # 1. Grid points
                grid_size = 20
                xs_grid = np.linspace(0, w, grid_size)
                ys_grid = np.linspace(0, h, grid_size)
                
                candidates = []
                for gx in xs_grid:
                    for gy in ys_grid:
                        candidates.append((gx, gy))
                        
                # 2. Random points
                for _ in range(100):
                    rx = np.random.uniform(0, w)
                    ry = np.random.uniform(0, h)
                    candidates.append((rx, ry))
                    
                # 3. Points near existing circles (offsets)
                for cx, cy, cr in circles:
                    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                        for dist in [cr * 2, cr * 2.5, cr * 3]:
                            nx = cx + dist * np.cos(angle)
                            ny = cy + dist * np.sin(angle)
                            if 0 <= nx <= w and 0 <= ny <= h:
                                candidates.append((nx, ny))
                
                # Evaluate each candidate
                for cx, cy in candidates:
                    # Max radius at this position
                    r_max = min(cx, w - cx, cy, h - cy)
                    
                    for ox, oy, orr in circles:
                        d = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                        r_max = min(r_max, d - orr)
                    
                    if r_max > best_r:
                        best_r = r_max
                        best_x = cx
                        best_y = cy
                        
                if best_r <= 1e-6:
                    # No space left? Should not happen with N=21 in 2x1 rect
                    # But if it does, break or place tiny circle
                    break
                    
                circles.append((best_x, best_y, best_r))
                
            if len(circles) < N:
                continue
                
            # Convert to array
            x0 = np.array([c[0] for c in circles])
            y0 = np.array([c[1] for c in circles])
            r0 = np.array([c[2] for c in circles])
            
            # Local Refinement using Nelder-Mead on penalized objective
            # Variables: [x1, y1, r1, x2, y2, r2, ...]
            
            def unpack(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                return x, y, r
            
            def penalized_objective(vars_):
                x, y, r = unpack(vars_)
                
                # Sum of radii (to maximize)
                sum_r = np.sum(r)
                
                # Penalties
                pen = 0.0
                
                # Wall constraints
                # x - r >= 0, w - x - r >= 0
                # y - r >= 0, h - y - r >= 0
                pen += np.sum(np.maximum(0, r - x)**2)
                pen += np.sum(np.maximum(0, r - (w - x))**2)
                pen += np.sum(np.maximum(0, r - y)**2)
                pen += np.sum(np.maximum(0, r - (h - y))**2)
                
                # Overlap constraints
                # dist(i,j) - ri - rj >= 0
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        violation = r[i] + r[j] - d
                        if violation > 0:
                            pen += violation**2
                
                # Radius positivity
                pen += np.sum(np.maximum(0, -r)**2)
                
                # Weight for penalty
                # Start with high weight, maybe reduce?
                # For now, fixed weight. Since we start valid, pen=0.
                # If we move, pen increases.
                weight = 100.0
                
                return -sum_r + weight * pen
            
            x0_full = np.zeros(3*N)
            for i in range(N):
                x0_full[3*i] = x0[i]
                x0_full[3*i+1] = y0[i]
                x0_full[3*i+2] = r0[i]
                
            try:
                result = minimize(
                    penalized_objective, 
                    x0_full, 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
                )
                
                # Check validity of result
                x, y, r = unpack(result.x)
                
                valid = True
                # Check walls
                if np.any(x < r - 1e-5) or np.any(x > w - r + 1e-5):
                    valid = False
                if np.any(y < r - 1e-5) or np.any(y > h - r + 1e-5):
                    valid = False
                if np.any(r <= 0):
                    valid = False
                    
                # Check overlaps
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < r[i] + r[j] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])
                        
            except Exception:
                continue
                
    if best_result is None:
        # Fallback: simple grid
        # 5x5 grid = 25 cells, use 21
        # w+h=2, let w=1, h=1
        w = 1.0
        h = 1.0
        n_cols = 5
        n_rows = 5
        r = 0.5 / n_cols - 0.01
        res = []
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            res.append([x, y, r])
        return np.array(res)
        
    return best_result
