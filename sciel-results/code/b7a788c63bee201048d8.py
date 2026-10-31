import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_circles = None
    
    # Objective function to minimize: -sum(r) + penalty
    def objective(params, W, H):
        # params shape: (N, 3) -> x, y, r
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Base objective: maximize sum of radii
        val = -np.sum(r)
        
        # Penalty for overlaps
        # Calculate pairwise distances
        # d_ij^2 = (x_i - x_j)^2 + (y_i - y_j)^2
        # Vectorized distance calculation
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        
        # Required distance: r_i + r_j
        req_d = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Violation: max(0, req_d - d)
        # Exclude diagonal (i==j)
        np.fill_diagonal(req_d, 0)
        np.fill_diagonal(d, np.inf)
        
        violations = np.maximum(0, req_d - d)
        overlap_penalty = np.sum(violations**2)
        
        # Penalty for boundary violations
        # x must be in [r, W-r] -> r <= x and r <= W-x
        # y must be in [r, H-r] -> r <= y and r <= H-y
        v_left = np.maximum(0, r - x)
        v_right = np.maximum(0, r - (W - x))
        v_bottom = np.maximum(0, r - y)
        v_top = np.maximum(0, r - (H - y))
        
        boundary_penalty = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Weight for penalties
        # We want constraints to be strong.
        # Since we are maximizing r, the optimizer will push r up.
        # A weight of 100-1000 usually works well for squared penalties.
        weight = 50.0
        val += weight * (overlap_penalty + boundary_penalty)
        
        return val

    def is_valid(circles, W, H):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check boundaries
        if np.any(r > x + 1e-6) or np.any(r > W - x + 1e-6) or np.any(r > y + 1e-6) or np.any(r > H - y + 1e-6):
            return False
        
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        
        req_d = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(req_d, 0)
        np.fill_diagonal(d, np.inf)
        
        # Allow tiny numerical error
        if np.any(req_d > d + 1e-6):
            return False
            
        return True

    def shrink_to_valid(circles, W, H):
        # If invalid, shrink radii until valid
        # This is a fallback
        for _ in range(100):
            if is_valid(circles, W, H):
                return circles
            # Shrink radii by 1%
            circles = circles.copy()
            circles[:, 2] *= 0.99
            if np.all(circles[:, 2] < 1e-5):
                break
        return circles

    # Aspect ratios to try
    # W + H = 2
    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        # Try a few restarts for this aspect ratio
        for restart in range(4):
            if time.time() > deadline:
                break
            
            # Initialization: Grid with jitter
            # 5x5 grid = 25 cells, use 21
            side = 5
            cells_x = (np.arange(side) + 0.5) / side
            cells_y = (np.arange(side) + 0.5) / side
            
            # Generate grid points
            grid_points = []
            for i in range(side):
                for j in range(side):
                    grid_points.append([i, j])
            
            # Shuffle to pick 21 distinct positions? Or just take first 21?
            # Taking first 21 creates a hole. Better to distribute.
            # Let's just take 21 random indices from the 25 grid points
            np.random.seed(42 + restart)
            indices = np.random.choice(25, N, replace=False)
            
            x0 = np.zeros((N, 3))
            for k, idx in enumerate(indices):
                i = idx // side
                j = idx % side
                # Add jitter
                x0[k, 0] = cells_x[i] * W + np.random.uniform(-0.01, 0.01)
                x0[k, 1] = cells_y[j] * H + np.random.uniform(-0.01, 0.01)
                x0[k, 2] = 0.1 # Initial radius
            
            # Bounds for L-BFGS-B
            # x in [0, W], y in [0, H], r in [0, 1]
            bounds = []
            for i in range(N):
                bounds.append((0.0, W)) # x
                bounds.append((0.0, H)) # y
                bounds.append((0.0, 1.0)) # r
            
            # Flatten params
            x0_flat = x0.flatten()
            
            # Optimize
            try:
                res = minimize(objective, x0_flat, args=(W, H), method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
                
                if res.success or res.fun < 0: # If it improved
                    x_opt = res.x.reshape(N, 3)
                    
                    # Check validity
                    if is_valid(x_opt, W, H):
                        score = np.sum(x_opt[:, 2])
                        if score > best_score:
                            best_score = score
                            best_circles = x_opt
                    else:
                        # Try to shrink
                        x_valid = shrink_to_valid(x_opt, W, H)
                        if is_valid(x_valid, W, H):
                            score = np.sum(x_valid[:, 2])
                            if score > best_score:
                                best_score = score
                                best_circles = x_valid
                            
            except Exception as e:
                pass

    # If no valid circles found, return a trivial valid solution
    if best_circles is None:
        # Grid solution
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([
                        (i + 0.5) / side,
                        (j + 0.5) / side,
                        0.5 / side - 0.01
                    ])
        best_circles = np.array(cells[:N])
        
    return best_circles
