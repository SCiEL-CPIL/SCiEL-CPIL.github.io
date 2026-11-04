import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial grid setup
    # Try a few different aspect ratios for the rectangle
    best_score = -np.inf
    best_result = None
    
    # We will optimize w (width), and x, y, r for each circle. h = 2 - w.
    # Variables: w, x1, y1, r1, ..., x21, y21, r21
    # Total 1 + 3*21 = 64 variables.
    
    def objective(v):
        w = v[0]
        h = 2.0 - w
        if h <= 0 or w <= 0:
            return 1e10
        circles = v[1:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Penalty for boundaries
        pen = 0.0
        # x bounds
        pen += np.sum(np.maximum(0, r - x)**2)
        pen += np.sum(np.maximum(0, x - (w - r))**2)
        # y bounds
        pen += np.sum(np.maximum(0, r - y)**2)
        pen += np.sum(np.maximum(0, y - (h - r))**2)
        # r bounds (should be positive)
        pen += np.sum(np.maximum(0, -r)**2)
        
        # Pairwise penalties
        for i in range(N):
            for j in range(i+1, N):
                dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                min_dist = r[i] + r[j]
                if dist2 < min_dist**2:
                    pen += (min_dist**2 - dist2)**2
        
        # We want to maximize sum of radii, so minimize negative sum
        # But we need to ensure constraints are satisfied.
        # A common trick is to return penalty - sum(r) * large_factor?
        # Or use SLSQP with explicit constraints.
        
        return pen - 10.0 * np.sum(r)

    def constraints(v):
        w = v[0]
        h = 2.0 - w
        circles = v[1:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        cons = []
        # w+h=2 is handled by variable definition
        # w > 0, h > 0
        # r > 0
        # x, y bounds
        # Instead of returning a list, we can use inequality constraints in SLSQP
        # But SLSQP expects a dict or list of dicts.
        # This might be slow to construct.
        # Let's stick to the penalty method in the objective for speed, 
        # but ensure the penalty is strong enough.
        # Actually, let's use 'L-BFGS-B' with penalty in objective, 
        # or 'SLSQP' with explicit constraints if possible.
        # Given N=21, 210 pairs. Explicit constraints might be too many for SLSQP speed.
        # Penalty method with L-BFGS-B is faster.
        
        return []

    def jac(v):
        # Numerical gradient is fine for 64 vars
        return None

    # Try a few initial aspect ratios
    for w_init in [1.0, 1.2, 0.8, 1.5, 0.5]:
        if time.time() > deadline:
            break
            
        h_init = 2.0 - w_init
        if h_init <= 0:
            continue
            
        # Initialize grid
        # Approximate grid for 21 circles
        # 5x5 grid, remove 4 corners? Or 3x7?
        # Let's use a dense grid initialization
        n_x = 5
        n_y = 5
        # Create grid points
        xs = np.linspace(w_init/(2*n_x), w_init - w_init/(2*n_x), n_x)
        ys = np.linspace(h_init/(2*n_y), h_init - h_init/(2*n_y), n_y)
        
        # Create all grid points
        grid_points = []
        for i in range(n_x):
            for j in range(n_y):
                grid_points.append([xs[i], ys[j]])
        
        # We need 21 points. Pick the 21 closest to the center?
        # Or just take the first 21?
        # Let's take the central ones.
        center_x = w_init / 2
        center_y = h_init / 2
        dists = [(p[0]-center_x)**2 + (p[1]-center_y)**2 for p in grid_points]
        sorted_indices = np.argsort(dists)
        
        init_circles = []
        for idx in sorted_indices[:N]:
            p = grid_points[idx]
            # Estimate radius based on spacing
            # Min distance to neighbors
            r_est = 0.1
            init_circles.append([p[0], p[1], r_est])
            
        v0 = np.array([w_init] + [val for c in init_circles for val in c])
        
        # Optimize
        try:
            res = minimize(objective, v0, method='L-BFGS-B', jac=jac, options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Check validity
                w = res.x[0]
                h = 2.0 - w
                circles = res.x[1:].reshape(N, 3)
                x = circles[:, 0]
                y = circles[:, 1]
                r = circles[:, 2]
                
                valid = True
                if w <= 0 or h <= 0:
                    valid = False
                for i in range(N):
                    if r[i] < 0:
                        valid = False
                    if x[i] < r[i] - 1e-5 or x[i] > w - r[i] + 1e-5:
                        valid = False
                    if y[i] < r[i] - 1e-5 or y[i] > h - r[i] + 1e-5:
                        valid = False
                
                if valid:
                    # Check pairwise
                    for i in range(N):
                        for j in range(i+1, N):
                            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                            min_dist = r[i] + r[j]
                            if dist2 < min_dist**2 - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        score = np.sum(r)
                        if score > best_score:
                            best_score = score
                            best_result = res.x.copy()
                            
        except Exception:
            pass

    if best_result is not None:
        w = best_result[0]
        h = 2.0 - w
        circles = best_result[1:].reshape(N, 3)
        # Ensure strict validity by shrinking radii slightly if needed
        # and clamping positions
        final_circles = []
        for i in range(N):
            x, y, r = circles[i]
            # Shrink r by 1e-4 to be safe
            r_new = max(0, r - 1e-4)
            # Clamp x, y
            x_new = np.clip(x, r_new, w - r_new)
            y_new = np.clip(y, r_new, h - r_new)
            final_circles.append([x_new, y_new, r_new])
            
        return np.array(final_circles)
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
