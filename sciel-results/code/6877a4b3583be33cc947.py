"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(x, y, r, eps=1e-9):
    """Check if a circle with center (x,y) and radius r is inside [0,1]^2."""
    return (r >= 0) and (x >= r - eps) and (x <= 1 - r + eps) and (y >= r - eps) and (y <= 1 - r + eps)

def _check_disjoint(circles, eps=1e-9):
    """Check if all circles have disjoint interiors."""
    n = len(circles)
    for i in range(n):
        xi, yi, ri = circles[i]
        if not _is_valid(xi, yi, ri, eps):
            return False
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - eps:
                return False
    return True

def _objective(params, N):
    """Objective function for optimization: minimize negative sum of radii."""
    # params: 3*N array: x1,y1,r1, x2,y2,r2, ...
    # We want to maximize sum(r_i) subject to constraints.
    # Using penalty method for constraints.
    
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Penalty for being outside [0,1]^2
    pen = 0.0
    for i in range(N):
        if r[i] < 0:
            pen += (r[i] + 1.0)**2 * 1000  # Strong penalty for negative radius
        if x[i] < r[i]:
            pen += (x[i] - r[i])**2 * 1000
        if x[i] > 1 - r[i]:
            pen += (x[i] - (1 - r[i]))**2 * 1000
        if y[i] < r[i]:
            pen += (y[i] - r[i])**2 * 1000
        if y[i] > 1 - r[i]:
            pen += (y[i] - (1 - r[i]))**2 * 1000
    
    # Penalty for overlapping circles
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                pen += (min_dist - dist)**2 * 1000
    
    # We want to maximize sum(r), so minimize -sum(r) + penalty
    return -np.sum(r) + pen

def _make_initial_grid(N):
    """Create an initial grid arrangement."""
    # Use a hexagonal-like grid for better packing
    side = int(np.ceil(np.sqrt(N)))
    # Start with a grid, then we'll optimize
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Simple grid initialization
    for i in range(N):
        row = i // side
        col = i % side
        x[i] = (col + 0.5) / side
        y[i] = (row + 0.5) / side
        r[i] = 0.5 / side - 1e-4
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    best_circles = None
    best_value = 0.0
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Create initial positions
        x, y, r = _make_initial_grid(N)
        
        # Add some randomness to break symmetry
        if attempt > 0:
            x = x + np.random.uniform(-0.1, 0.1, N)
            y = y + np.random.uniform(-0.1, 0.1, N)
            # Clip to valid range
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
        
        params = np.concatenate([x, y, r])
        
        # Optimize
        try:
            result = minimize(
                _objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception as e:
            continue
        
        if result.success or result.status == 0:
            opt_params = result.x
            opt_x = opt_params[:N]
            opt_y = opt_params[N:2*N]
            opt_r = opt_params[2*N:3*N]
            
            # Ensure radii are non-negative
            opt_r = np.maximum(opt_r, 0)
            
            # Check validity
            circles = list(zip(opt_x, opt_y, opt_r))
            
            # Post-process: ensure circles are inside and disjoint
            # If invalid, try to fix by shrinking radii
            valid = _check_disjoint(circles, eps=1e-8)
            
            if valid:
                value = np.sum(opt_r)
                if value > best_value:
                    best_value = value
                    best_circles = circles
            
            # Even if not perfectly valid, try to use it by shrinking
            # to make it valid
            if not valid:
                # Try to make valid by reducing radii
                shrunk_circles = []
                for i, (xi, yi, ri) in enumerate(circles):
                    # Check boundaries
                    max_r = min(xi, 1 - xi, yi, 1 - yi)
                    ri = min(ri, max_r)
                    ri = max(ri, 0)
                    
                    # Check overlaps with already placed circles
                    for xj, yj, rj in shrunk_circles:
                        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                        max_r = min(max_r, dist - rj)
                    ri = min(ri, max(0, max_r))
                    
                    shrunk_circles.append((xi, yi, ri))
                
                if _check_disjoint(shrunk_circles, eps=1e-8):
                    value = np.sum([c[2] for c in shrunk_circles])
                    if value > best_value:
                        best_value = value
                        best_circles = shrunk_circles
        
        # Also try with different random seeds for diversity
        np.random.seed(42 + attempt * 1000)
    
    # If no valid solution found, fall back to grid
    if best_circles is None:
        x, y, r = _make_initial_grid(N)
        best_circles = list(zip(x, y, r))
        best_value = np.sum(r)
    
    return np.array(best_circles)
# EVOLVE-BLOCK-END
