"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def construct():
    """
    Pack N=21 circles in a rectangle of perimeter <= 4 to maximize sum of radii.
    
    We use a grid-based initial guess and then perform local optimization.
    The rectangle has width w and height h with 2*(w+h) <= 4, i.e., w+h <= 2.
    We can fix the aspect ratio or optimize over it. For simplicity, we'll 
    start with a square (w=h=1) and then try to improve.
    
    Actually, let's think about this more carefully. The perimeter constraint is 2(w+h) <= 4,
    so w + h <= 2. We want to maximize sum(r_i) subject to:
    - Circles are inside the rectangle: x_i - r_i >= 0, x_i + r_i <= w, y_i - r_i >= 0, y_i + r_i <= h
    - Circles don't overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    - w + h <= 2
    
    This is a non-convex problem. We'll use a local optimization approach with a good initial guess.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial setup: try a grid arrangement
    # For 21 circles, a 5x5 grid with some removed, or 4x6, etc.
    # Let's try different grid configurations
    
    best_value = 0
    best_solution = None
    
    # Try different aspect ratios and grid arrangements
    # We'll parameterize by aspect ratio r = w/h
    # w = 2*r/(1+r), h = 2/(1+r) for w+h=2
    
    # First, let's try a simple grid-based approach with optimization
    
    def evaluate_solution(w, h, circles):
        """Check if solution is valid and return sum of radii."""
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check boundary constraints
        if np.any(x - r < -1e-6) or np.any(x + r > w + 1e-6):
            return -np.inf
        if np.any(y - r < -1e-6) or np.any(y + r > h + 1e-6):
            return -np.inf
        
        # Check overlap
        n = len(r)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-6:
                    return -np.inf
        
        # Check perimeter
        if 2*(w + h) > 4 + 1e-6:
            return -np.inf
        
        return np.sum(r)
    
    def grid_initial_guess(n_cols, n_rows, w, h):
        """Create a grid-based initial guess."""
        circles = []
        for i in range(n_cols):
            for j in range(n_rows):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * w / n_cols
                cy = (j + 0.5) * h / n_rows
                # Max radius is limited by spacing and boundaries
                r_max = min(w/n_cols, h/n_rows) / 2
                r = r_max * 0.95  # Slightly smaller to avoid touching
                circles.append([cx, cy, r])
        return np.array(circles[:N])
    
    def optimize_circles(circles_init, w, h, max_iter=1000):
        """Optimize circle positions and radii given fixed w, h."""
        n = len(circles_init)
        
        def objective(params):
            x = params[0:n]
            y = params[n:2*n]
            r = params[2*n:3*n]
            
            # Boundary constraints (soft penalty)
            penalty = 0
            for i in range(n):
                if x[i] - r[i] < 0:
                    penalty += (x[i] - r[i])**2
                if x[i] + r[i] > w:
                    penalty += (x[i] + r[i] - w)**2
                if y[i] - r[i] < 0:
                    penalty += (y[i] - r[i])**2
                if y[i] + r[i] > h:
                    penalty += (y[i] + r[i] - h)**2
            
            # Overlap constraints (soft penalty)
            for i in range(n):
                for j in range(i+1, n):
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    min_dist = r[i] + r[j]
                    if dist_sq < min_dist**2:
                        penalty += (min_dist - np.sqrt(dist_sq))**2
            
            return -np.sum(r) + 100 * penalty
        
        # Initial parameters
        x0 = np.concatenate([circles_init[:, 0], circles_init[:, 1], circles_init[:, 2]])
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((0, w))  # x
        for i in range(n):
            bounds.append((0, h))  # y
        for i in range(n):
            bounds.append((1e-4, min(w, h)/2))  # r
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success:
            x = result.x[0:n]
            y = result.x[n:2*n]
            r = result.x[2*n:3*n]
            circles = np.column_stack([x, y, r])
            val = evaluate_solution(w, h, circles)
            if val > 0:
                return circles, val
            else:
                return None, 0
        else:
            return None, 0
    
    # Try several grid configurations
    configs = [
        (5, 5), (4, 6), (6, 4), (7, 3), (3, 7), (5, 4), (4, 5), (6, 3), (3, 6),
        (7, 4), (4, 7), (8, 3), (3, 8), (9, 3), (3, 9), (10, 3), (3, 10),
        (11, 2), (2, 11), (12, 2), (2, 12), (14, 2), (2, 14), (21, 1), (1, 21)
    ]
    
    for n_cols, n_rows in configs:
        if time.time() > deadline - 10:
            break
        
        # Try different aspect ratios
        for aspect in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
            if time.time() > deadline - 5:
                break
            
            # Calculate w and h from aspect ratio and w+h=2
            w = 2 * aspect / (1 + aspect)
            h = 2 / (1 + aspect)
            
            # Check if grid fits
            if n_cols * w / n_cols < 0.01 or n_rows * h / n_rows < 0.01:
                continue
            
            circles_init = grid_initial_guess(n_cols, n_rows, w, h)
            if len(circles_init) < N:
                continue
            
            circles, val = optimize_circles(circles_init, w, h, max_iter=500)
            
            if circles is not None and val > best_value:
                best_value = val
                best_solution = circles
    
    # If we found a solution, return it; otherwise, return a simple valid solution
    if best_solution is not None:
        return best_solution
    else:
        # Fallback: simple grid in a 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
