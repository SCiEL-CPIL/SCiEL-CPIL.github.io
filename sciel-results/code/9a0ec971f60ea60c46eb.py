import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 32

def construct():
    """
    Pack N circles in the unit square [0, 1]^2 to maximize the sum of radii.
    Returns an array of shape (N, 3) with rows (x, y, r).
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    # Initial guess: Grid layout with jitter
    side = int(np.ceil(np.sqrt(N)))
    # Create a grid of points
    grid_x = (np.arange(side) + 0.5) / side
    grid_y = (np.arange(side) + 0.5) / side
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    # Flatten and take first N
    init_x = xx.flatten()[:N]
    init_y = yy.flatten()[:N]
    
    # Initial radii: slightly less than half the grid cell size to avoid initial overlap
    # Grid cell size is 1/side. Max radius approx 0.5/side.
    init_r = np.full(N, 0.5 / side - 0.01)
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    jitter = 0.01
    init_x += np.random.uniform(-jitter, jitter, N)
    init_y += np.random.uniform(-jitter, jitter, N)
    
    # Ensure initial positions are within bounds
    init_x = np.clip(init_x, 0, 1)
    init_y = np.clip(init_y, 0, 1)
    
    # Flatten to 1D array for optimizer: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    x0[0::3] = init_x
    x0[1::3] = init_y
    x0[2::3] = init_r
    
    # Objective function: Minimize negative sum of radii + penalties
    # We want to maximize sum(r), so minimize -sum(r) + penalty
    def objective(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Penalty for going out of bounds
        # x must be in [r, 1-r] => x - r >= 0 and 1 - x - r >= 0
        # Violation is max(0, r - x) + max(0, r - (1 - x))
        bound_pen_x = np.sum(np.maximum(0, r - x)) + np.sum(np.maximum(0, r - (1 - x)))
        bound_pen_y = np.sum(np.maximum(0, r - y)) + np.sum(np.maximum(0, r - (1 - y)))
        
        # Penalty for overlap
        # Distance between i and j: sqrt((xi-xj)^2 + (yi-yj)^2)
        # Overlap if dist < ri + rj
        # Penalty term: max(0, ri + rj - dist)^2
        overlap_pen = 0.0
        # Vectorized calculation of pairwise distances is efficient for N=32
        # Diff matrix
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Sum of radii for each pair
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Violation amount
        viol = np.maximum(0, r_sum - dist)
        
        # Square the violation and sum
        overlap_pen = np.sum(viol**2)
        
        # Total penalty
        # Large multiplier to ensure constraints are satisfied
        K = 1000.0
        penalty = K * (bound_pen_x + bound_pen_y + overlap_pen)
        
        return -sum_r + penalty

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius in unit square)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r

    best_result = None
    best_score = -np.inf
    
    # Try a few different random seeds/jitters if time allows, but start with the main one
    # Since L-BFGS-B is deterministic given x0, we just run it once with the jittered start.
    # If we have time, we could try different jitter magnitudes or seeds.
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or (res.fun < best_score):
            best_result = res.x
            best_score = res.fun
            
        # Check time
        if time.time() > deadline:
            pass # Just exit loop if we had multiple, but here we only do one main run
            
    except Exception as e:
        # If optimization fails, fall back to initial grid
        pass

    # If optimization failed or we didn't get a result, use the initial grid
    if best_result is None:
        v = x0
    else:
        v = best_result

    # Extract final coordinates
    x_final = v[0::3]
    y_final = v[1::3]
    r_final = v[2::3]
    
    # Post-processing: Ensure strict feasibility
    # Clip x, y to [r, 1-r]
    # This might slightly reduce radii if they were too big, but ensures validity
    # Actually, if the optimizer worked, they should be feasible.
    # But to be safe against floating point errors:
    for i in range(N):
        # Adjust radius if it violates bounds
        max_r_x = min(x_final[i], 1 - x_final[i])
        max_r_y = min(y_final[i], 1 - y_final[i])
        r_final[i] = min(r_final[i], max_r_x, max_r_y)
        
        # If radius became 0 or negative, set to small epsilon
        if r_final[i] < 1e-6:
            r_final[i] = 1e-6
    
    # Check pairwise overlaps one last time and shrink if necessary
    # This is a simple repair step
    for _ in range(10): # Iterate a few times to resolve overlaps
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x_final[i] - x_final[j]
                dy = y_final[i] - y_final[j]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = r_final[i] + r_final[j]
                if dist < r_sum:
                    # Shrink both radii slightly to fix overlap
                    # Reduce each by half the overlap amount
                    overlap = (r_sum - dist) / 2.0
                    r_final[i] = max(1e-6, r_final[i] - overlap)
                    r_final[j] = max(1e-6, r_final[j] - overlap)
                    changed = True
        if not changed:
            break
            
    # Construct output array
    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
