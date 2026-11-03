"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    N = 21
    # Perimeter 4 => w + h = 2.
    # We will optimize for a rectangle of fixed aspect ratio or variable?
    # The problem says "variable rectangle of perimeter 4".
    # We can fix w and h to a good aspect ratio, e.g., w=1.4, h=0.6 (3x7 grid)
    # or optimize w and h as well.
    # Let's fix w=1.4, h=0.6 for simplicity and robustness, as 3x7 is a good start.
    # Or better, let's try a few aspect ratios.
    
    best_result = None
    best_value = -np.inf

    # Helper to validate and fix a packing
    def validate_and_fix(circles, w, h):
        # circles: (N, 3) array of [x, y, r]
        # Ensure within bounds
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
        
        # Ensure no overlaps by shrinking radii if necessary
        # This is a simple heuristic: if overlap, reduce radii of the overlapping pair
        # A more robust way is to solve a constraint satisfaction problem, but for speed:
        # We'll just check and if invalid, return None.
        # However, we want to return the best valid result.
        
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check bounds strictly
            if np.all(circles[:, 0] >= 0) and np.all(circles[:, 0] <= w) and \
               np.all(circles[:, 1] >= 0) and np.all(circles[:, 1] <= h) and \
               np.all(circles[:, 2] > 0):
                return circles
        return None

    # Strategy 1: Staggered Grid (Hexagonal packing approximation)
    # For 3 rows, 7 cols is 21 circles.
    # Row 0: 7 circles
    # Row 1: 7 circles (shifted)
    # Row 2: 7 circles
    # This fits in a 3x7 grid.
    
    from scipy.optimize import minimize
    
    # Try different aspect ratios
    aspect_ratios = [1.4/0.6, 1.0, 1.2/0.8] # w/h
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        # w + h = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initialize with staggered grid
        # 3 rows, 7 cols
        # Row height = h/3
        # Col width = w/7
        # Radius approx = min(w/7, h/3) / 2 * 0.95 (shrink to allow movement)
        
        rows = 3
        cols = 7
        row_h = h / rows
        col_w = w / cols
        
        x0 = np.zeros(N * 3)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                # Center x
                cx = (j + 0.5) * col_w
                # Stagger: shift odd rows by half cell width
                if i % 2 == 1:
                    cx += col_w / 2.0
                # Ensure within bounds
                cx = np.clip(cx, 0.1, w - 0.1)
                
                cy = (i + 0.5) * row_h
                
                # Initial radius
                r = min(col_w, row_h) * 0.45 # 0.45 instead of 0.5 to allow optimization
                
                x0[idx*3] = cx
                x0[idx*3+1] = cy
                x0[idx*3+2] = r
                idx += 1
        
        # Objective function
        def objective(x):
            positions = x.reshape(N, 3)
            cxs = positions[:, 0]
            cys = positions[:, 1]
            rs = positions[:, 2]
            
            # Penalty for going out of bounds
            pen = 0.0
            for i in range(N):
                if cxs[i] < rs[i]:
                    pen += (rs[i] - cxs[i])**2
                if cxs[i] > w - rs[i]:
                    pen += (cxs[i] - (w - rs[i]))**2
                if cys[i] < rs[i]:
                    pen += (rs[i] - cys[i])**2
                if cys[i] > h - rs[i]:
                    pen += (cys[i] - (h - rs[i]))**2
            
            # Penalty for overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = cxs[i] - cxs[j]
                    dy = cys[i] - cys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        pen += (min_dist - dist)**2
            
            # Maximize sum of radii => Minimize -sum(rs) + penalty
            return -np.sum(rs) + 10.0 * pen
        
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0.01, 0.5))
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.fun < -0.1: # Valid improvement
                final_circles = res.x.reshape(N, 3)
                # Validate
                valid_circles = validate_and_fix(final_circles, w, h)
                if valid_circles is not None:
                    val = np.sum(valid_circles[:, 2])
                    if val > best_value:
                        best_value = val
                        best_result = valid_circles.copy()
        except Exception:
            pass

    # If optimization failed or didn't find anything better than a simple grid,
    # use the simple 3x7 grid with slightly reduced radius to ensure validity.
    if best_result is None or best_value < 2.0:
        w, h = 1.4, 0.6
        cells = []
        for i in range(3):
            for j in range(7):
                cx = (j + 0.5) * w / 7
                cy = (i + 0.5) * h / 3
                # Use 0.0999 to be strictly non-overlapping
                cells.append([cx, cy, 0.0999])
        best_result = np.array(cells)
        best_value = np.sum(best_result[:, 2])

    return best_result
# EVOLVE-BLOCK-END
