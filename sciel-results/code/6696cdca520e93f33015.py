import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Candidate aspect ratios (W, H=2-W)
    # W ranges from 0.5 to 1.5
    w_candidates = np.linspace(0.7, 1.3, 13)
    
    # Initial layout generators
    def make_grid_layout(n_x, n_y, W, H):
        # Returns (x, y) for n_x * n_y points, taking first N
        # We need to fill N points.
        # If n_x * n_y > N, we take the first N.
        # This creates a "missing" corner, which might be suboptimal.
        # Better to center the grid.
        x_coords = []
        y_coords = []
        total = n_x * n_y
        count = 0
        for i in range(n_x):
            for j in range(n_y):
                if count >= N:
                    break
                # Center the grid
                x = (i + 0.5) * (W / n_x)
                y = (j + 0.5) * (H / n_y)
                x_coords.append(x)
                y_coords.append(y)
                count += 1
            if count >= N:
                break
        return np.array(x_coords), np.array(y_coords)

    def make_hex_layout(n_x, n_y, W, H):
        # Hexagonal packing
        # Row 0: n_x circles
        # Row 1: n_x - 1 or n_x circles (shifted)
        # Let's just use a grid for simplicity in initialization, 
        # the optimizer will fix it.
        return make_grid_layout(n_x, n_y, W, H)

    def objective(params, W, H):
        # params: x1..xN, y1..yN
        x = params[:N]
        y = params[N:2*N]
        
        # Distances to walls
        dx_min = np.minimum(x, W - x)
        dy_min = np.minimum(y, H - y)
        wall_dists = np.minimum(dx_min, dy_min)
        
        # Distances between circles
        # Calculate pairwise distances
        # To avoid N^2 loop in Python, use numpy broadcasting
        # But N=21, so 21x21 matrix is small.
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        # The minimum distance is the minimum of wall dists and pair dists
        min_dist = np.min([np.min(wall_dists), min_pair_dist])
        
        # We want to maximize min_dist, so minimize -min_dist
        return -min_dist

    # Try different grid configurations for initialization
    configs = [
        (5, 5),
        (6, 4),
        (7, 3),
        (3, 7),
        (4, 6),
        (9, 3), # 27 cells
        (7, 4), # 28 cells
    ]
    
    for W in w_candidates:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        for n_x, n_y in configs:
            if time.time() > deadline:
                break
            
            x0, y0 = make_grid_layout(n_x, n_y, W, H)
            # If we didn't get N points, pad with random or center
            if len(x0) < N:
                # This shouldn't happen with the configs above
                continue
            
            x0 = x0[:N]
            y0 = y0[:N]
            
            params0 = np.concatenate([x0, y0])
            
            # Bounds for x and y
            # x in [0, W], y in [0, H]
            bounds = [(0, W)] * N + [(0, H)] * N
            
            # Run optimization
            # Nelder-Mead is good for this
            try:
                res = minimize(objective, params0, args=(W, H), method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.success or res.fun < best_score * -1: # res.fun is -min_dist
                    min_dist = -res.fun
                    # Score is sum of radii = N * (min_dist / 2)
                    score = N * min_dist / 2.0
                    
                    if score > best_score:
                        best_score = score
                        x_opt = res.x[:N]
                        y_opt = res.x[N:2*N]
                        r_opt = min_dist / 2.0
                        best_result = np.column_stack((x_opt, y_opt, np.full(N, r_opt)))
            except Exception:
                pass

    if best_result is None:
        # Fallback to seed-like behavior
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
