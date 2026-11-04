"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _objective(x, W, H):
    """
    Objective function to minimize: -sum(radii) + penalty for violations.
    x = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = N
    centers = x[0:2*n:3]
    radii = x[2:3*n:3]
    
    # Penalty for circles going out of bounds
    penalty = 0.0
    for i in range(n):
        r = radii[i]
        cx = centers[i]
        cy = x[2*i+1]
        
        # Left
        if cx < r:
            penalty += (r - cx) ** 2
        # Right
        if cx > W - r:
            penalty += (cx - (W - r)) ** 2
        # Bottom
        if cy < r:
            penalty += (r - cy) ** 2
        # Top
        if cy > H - r:
            penalty += (cy - (H - r)) ** 2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i] - centers[j]
            dy = x[2*i+1] - x[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    # Return negative sum of radii plus penalty (we want to minimize)
    return -np.sum(radii) + 10.0 * penalty


def _init_grid(n_circles, W, H):
    """Initialize circles on a grid within rectangle W x H."""
    # Use a 5x5 grid for 21 circles (last row has 1)
    cols = 5
    rows = 5
    
    dx = W / cols
    dy = H / rows
    
    centers = []
    radii = []
    
    for row in range(rows):
        for col in range(cols):
            if len(centers) < n_circles:
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                r = min(dx, dy) / 2.0 * 0.8  # 80% of max to allow optimization
                centers.append(x)
                radii.append(r)
    
    # Build x vector: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    x = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x[3*i] = centers[i]
        x[3*i+1] = (i // cols + 0.5) * dy
        x[3*i+2] = radii[i]
    
    return x


def _optimize_aspect(aspect, time_budget):
    """
    Optimize packing for given aspect ratio using gradient-based optimization.
    Returns (score, x_opt) or (None, None).
    """
    W = 2.0 / (1.0 + aspect)
    H = 2.0 * aspect / (1.0 + aspect)
    
    # Initialize
    x0 = _init_grid(N, W, H)
    
    # Optimize
    start_time = time.time()
    
    try:
        result = minimize(
            _objective,
            x0,
            args=(W, H),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return None, None
        
        x_opt = result.x
        radii = x_opt[2:3*N:3]
        score = np.sum(radii)
        
        # Verify validity
        centers_x = x_opt[0:2*N:3]
        centers_y = x_opt[1:3*N:3]
        
        valid = True
        for i in range(N):
            r = radii[i]
            cx = centers_x[i]
            cy = centers_y[i]
            
            if cx < -1e-6 or cx > W + 1e-6 or cy < -1e-6 or cy > H + 1e-6:
                valid = False
                break
            if r < -1e-6:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers_x[i] - centers_x[j]
                    dy = centers_y[i] - centers_y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if not valid:
            # Try to fix by scaling down radii slightly
            scale = 0.95
            for i in range(N):
                x_opt[3*i+2] *= scale
            score = np.sum(x_opt[2:3*N:3])
        
        return score, x_opt
    
    except Exception:
        return None, None


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = 0.0
    best_x = None
    
    # Try multiple aspect ratios
    aspects = [0.8, 0.9, 1.0, 1.1, 1.25]
    
    for aspect in aspects:
        elapsed = time.time() - start_time
        time_budget = max(1.0, deadline - elapsed - 5.0)
        if time_budget <= 0:
            break
        
        score, x_opt = _optimize_aspect(aspect, time_budget)
        
        if score is not None and score > best_score:
            best_score = score
            best_x = x_opt
    
    # Fallback if nothing worked
    if best_x is None:
        W = 1.0
        H = 1.0
        x0 = _init_grid(N, W, H)
        best_x = x0
    
    return best_x.reshape(N, 3)

# EVOLVE-BLOCK-END
