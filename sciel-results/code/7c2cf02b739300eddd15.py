import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None

    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ...]
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Maximize sum of radii -> Minimize negative sum
        obj = -np.sum(r)
        
        # Penalty for overlaps
        # Vectorized distance calculation
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Pairwise sum of radii
        r_sum = r[:, None] + r[None, :]
        
        # Overlap amount (positive if overlapping)
        overlap = np.maximum(0, r_sum - dist)
        # Zero out diagonal
        np.fill_diagonal(overlap, 0)
        
        overlap_penalty = np.sum(overlap**2)
        
        # Penalty for being outside the square
        # x < r, x > 1-r, y < r, y > 1-r
        out_x_left = np.maximum(0, r - x)
        out_x_right = np.maximum(0, x - (1 - r))
        out_y_left = np.maximum(0, r - y)
        out_y_right = np.maximum(0, y - (1 - r))
        
        boundary_penalty = np.sum(out_x_left**2 + out_x_right**2 + out_y_left**2 + out_y_right**2)
        
        # Weights
        w_overlap = 100.0
        w_boundary = 100.0
        
        return obj + w_overlap * overlap_penalty + w_boundary * boundary_penalty

    def hex_initialization(scale=1.0, offset_x=0.0, offset_y=0.0):
        # Generate a hexagonal grid of points
        # Adjust scale to fit N points roughly in [0,1]
        # For N=26, a 5x5 grid is 25, 6x5 is 30.
        # Let's try a grid that is slightly larger and pick the best N or just use a dense hex grid and pick N.
        # Actually, for optimization, we need exactly N variables.
        # Let's create a hex grid with enough points and select N, or just define N points in a hex-like pattern.
        
        # Simple approach: Create a grid of side L, place N points in hex pattern.
        # If N is not a perfect hex number, we just take the first N from a larger hex grid.
        
        # Let's generate a hex grid covering [0,1]x[0,1] with some density
        # and pick N points.
        
        # A better way: Start with a square grid of size x size, then perturb.
        # Or, use the known optimal packing structure if we knew it, but we don't.
        
        # Let's generate a hex lattice
        a = 0.2 # spacing
        rows = int(1.0 / (a * np.sqrt(3)/2)) + 2
        cols = int(1.0 / a) + 2
        
        pts = []
        for i in range(rows):
            for j in range(cols):
                x = j * a + (i % 2) * a / 2
                y = i * a * np.sqrt(3) / 2
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append((x, y))
        
        # Shuffle and pick N
        np.random.shuffle(pts)
        if len(pts) < N:
             # Fallback to random
             pts = [(np.random.rand(), np.random.rand()) for _ in range(N)]
        else:
             pts = pts[:N]
        
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        # Initial radii small to ensure validity
        r = np.full(N, 0.05)
        
        return np.concatenate([x, y, r]).reshape(-1, 3).T.flatten()

    def random_initialization():
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.random.rand(N) * 0.1 + 0.01
        return np.concatenate([x, y, r]).reshape(-1, 3).T.flatten()

    def grid_initialization():
        side = int(np.ceil(np.sqrt(N)))
        x = (np.arange(side) + 0.5) / side
        y = (np.arange(side) + 0.5) / side
        xx, yy = np.meshgrid(x, y)
        x = xx.flatten()
        y = yy.flatten()
        x = x[:N]
        y = y[:N]
        r = np.full(N, 0.1 / side)
        return np.concatenate([x, y, r]).reshape(-1, 3).T.flatten()

    # Bounds for variables:
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius in unit square is 0.5)
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    initializers = [hex_initialization, random_initialization, grid_initialization]
    
    # Multi-start
    for i in range(20): # Try 20 restarts
        if time.time() > deadline:
            break
            
        if i < len(initializers):
            init = initializers[i]()
        else:
            # Use perturbed versions of the best found so far, or random
            if best_solution is not None:
                init = best_solution + np.random.normal(0, 0.05, size=3*N)
                # Clip to bounds
                init[0::3] = np.clip(init[0::3], 0, 1)
                init[1::3] = np.clip(init[1::3], 0, 1)
                init[2::3] = np.clip(init[2::3], 0, 0.5)
            else:
                init = random_initialization()

        try:
            res = minimize(objective, init, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success or res.status == 0:
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                
                # Check validity
                valid = True
                # Boundary check
                if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6) or \
                   np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6):
                    valid = False
                # Overlap check
                if valid:
                    dx = x[:, None] - x[None, :]
                    dy = y[:, None] - y[None, :]
                    dist = np.sqrt(dx**2 + dy**2)
                    r_sum = r[:, None] + r[None, :]
                    np.fill_diagonal(r_sum, 0)
                    np.fill_diagonal(dist, np.inf)
                    if np.any(dist < r_sum - 1e-6):
                        valid = False
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_solution = res.x
                        # print(f"New best: {score:.4f} at iter {i}")

        except Exception as e:
            pass

    if best_solution is None:
        # Fallback to a very safe small grid
        side = int(np.ceil(np.sqrt(N)))
        x = (np.arange(side) + 0.5) / side
        y = (np.arange(side) + 0.5) / side
        xx, yy = np.meshgrid(x, y)
        x = xx.flatten()[:N]
        y = yy.flatten()[:N]
        r = np.full(N, 0.1 / side)
        best_solution = np.concatenate([x, y, r]).reshape(-1, 3).T.flatten()

    # Format output
    x = best_solution[0::3]
    y = best_solution[1::3]
    r = best_solution[2::3]
    
    return np.column_stack((x, y, r))

