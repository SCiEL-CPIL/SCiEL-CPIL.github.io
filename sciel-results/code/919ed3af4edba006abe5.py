import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate initial hexagonal arrangement
    # We want 21 circles.
    # Let's create a hex grid.
    # Rows: 5, 4, 5, 4, 3 = 21?
    # Or 4, 5, 4, 4, 4 = 21?
    # Let's try a compact cluster.
    # Hex coordinates:
    # x = i * sqrt(3) + (j % 2) * sqrt(3)/2
    # y = j * 1.5
    
    # Generate a grid of points
    points = []
    for j in range(6):
        for i in range(6):
            x = i * np.sqrt(3) + (j % 2) * np.sqrt(3) / 2
            y = j * 1.5
            points.append((x, y))
    
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Sort by distance to centroid and take N closest
    points.sort(key=lambda p: (p[0]-centroid[0])**2 + (p[1]-centroid[1])**2)
    initial_points = np.array(points[:N])
    
    # Center the points at origin
    initial_points -= np.mean(initial_points, axis=0)
    
    # Calculate bounding box
    min_x, min_y = np.min(initial_points, axis=0)
    max_x, max_y = np.max(initial_points, axis=0)
    w_init = max_x - min_x
    h_init = max_y - min_y
    
    # We need w + h = 2.
    # Scale the points to fit in a box of aspect ratio w_init/h_init
    # Let w = 2 * w_init / (w_init + h_init)
    # h = 2 * h_init / (w_init + h_init)
    w_init_scaled = 2.0 * w_init / (w_init + h_init)
    h_init_scaled = 2.0 * h_init / (w_init + h_init)
    
    # Scale points to fit in [0, w] x [0, h]
    # Current span is w_init, h_init.
    # New span should be slightly less than w, h to allow for radii.
    # Let's scale by 0.8 to leave room.
    scale_x = (w_init_scaled * 0.8) / w_init if w_init > 0 else 1
    scale_y = (h_init_scaled * 0.8) / h_init if h_init > 0 else 1
    
    # Apply scaling
    initial_points[:, 0] *= scale_x
    initial_points[:, 1] *= scale_y
    
    # Shift to center in [0, w] x [0, h]
    initial_points[:, 0] += w_init_scaled / 2
    initial_points[:, 1] += h_init_scaled / 2
    
    # Initial radii: small
    r_init = 0.1 * np.ones(N)
    
    # Variables: x1, y1, r1, ..., xN, yN, rN, w
    # h is derived: h = 2 - w
    
    def objective(v):
        if time.time() > deadline:
            return np.inf
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        # Penalty for w, h being positive and reasonable
        if w <= 0.1 or h <= 0.1:
            return 1e6
        
        cost = -np.sum(r)
        
        # Overlap penalty
        # Vectorized distance calculation
        # diff = x[:, None] - x[None, :]
        # dist = np.sqrt(diff**2 + (y[:, None] - y[None, :])**2)
        # For N=21, loops are fine and might be faster due to overhead of large arrays in Python if not careful, 
        # but numpy is usually faster. Let's use numpy.
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Set diagonal to infinity to ignore self
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist, axis=1) # This is not quite right for pairwise, we need all pairs
        
        # Better:
        # Create upper triangle mask
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        dist_pairs = dist[mask]
        r_sum_pairs = (r[:, None] + r[None, :])[mask]
        
        overlaps = np.maximum(0, r_sum_pairs - dist_pairs)
        overlap_penalty = np.sum(overlaps**2)
        
        # Boundary penalty
        # x >= r, w - x >= r, y >= r, h - y >= r
        bound_pen = 0
        bound_pen += np.sum(np.maximum(0, r - x)**2)
        bound_pen += np.sum(np.maximum(0, r - (w - x))**2)
        bound_pen += np.sum(np.maximum(0, r - y)**2)
        bound_pen += np.sum(np.maximum(0, r - (h - y))**2)
        
        # Weights
        w_overlap = 1000.0
        w_bound = 1000.0
        
        cost += w_overlap * overlap_penalty
        cost += w_bound * bound_pen
        
        return cost
    
    # Initial vector
    v0 = np.concatenate([initial_points[:, 0], initial_points[:, 1], r_init, [w_init_scaled]])
    
    # Bounds
    # x, y: [0, 2] (loose)
    # r: [1e-6, 1.0]
    # w: [0.1, 1.9]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((1e-6, 1.0)) # r
    bounds.append((0.1, 1.9)) # w
    
    best_score = -np.inf
    best_v = v0
    
    # Try multiple restarts with different aspect ratios
    # Aspect ratios: square, wide, tall
    aspect_ratios = [1.0, 1.5, 0.7, 2.0, 0.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        # Adjust initial w based on aspect ratio
        # w / h = ar => w / (2 - w) = ar => w = ar * (2 - w) => w(1+ar) = 2ar => w = 2ar/(1+ar)
        w_try = 2.0 * ar / (1.0 + ar)
        
        # Re-initialize points to fit this aspect ratio roughly
        # Just use the same points but scale x or y to match aspect?
        # Simpler: just change the initial w in v0.
        v_try = v0.copy()
        v_try[3*N] = w_try
        
        # Add some noise
        noise = 0.01 * np.random.randn(N, 2)
        v_try[0:N] += noise[:, 0]
        v_try[N:2*N] += noise[:, 1]
        
        # Clip to bounds
        for i in range(len(bounds)):
            if v_try[i] < bounds[i][0]:
                v_try[i] = bounds[i][0] + 1e-5
            if v_try[i] > bounds[i][1]:
                v_try[i] = bounds[i][1] - 1e-5
        
        try:
            res = minimize(objective, v_try, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            
            if time.time() > deadline:
                break
                
            if res.success or res.fun < objective(v_try):
                # Calculate score
                x = res.x[0:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                w = res.x[3*N]
                h = 2.0 - w
                
                # Check validity
                valid = True
                if w <= 0 or h <= 0:
                    valid = False
                else:
                    # Check overlaps
                    dx = x[:, None] - x[None, :]
                    dy = y[:, None] - y[None, :]
                    dist = np.sqrt(dx**2 + dy**2)
                    np.fill_diagonal(dist, np.inf)
                    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                    dist_pairs = dist[mask]
                    r_sum_pairs = (r[:, None] + r[None, :])[mask]
                    if np.any(dist_pairs < r_sum_pairs - 1e-5):
                        valid = False
                    # Check boundaries
                    if np.any(x < r - 1e-5) or np.any(w - x < r - 1e-5) or \
                       np.any(y < r - 1e-5) or np.any(h - y < r - 1e-5):
                        valid = False
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_v = res.x.copy()
                        
        except Exception:
            pass

    # If no valid solution found, return a safe fallback (small circles)
    if best_score == -np.inf:
        # Fallback: small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
        # Scale to perimeter 4 (w+h=2)
        # Assume square 1x1
        return np.array(cells[:N])

    # Extract best solution
    x = best_v[0:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    w = best_v[3*N]
    h = 2.0 - w
    
    # Construct output array (x, y, r)
    result = np.column_stack((x, y, r))
    
    return result
