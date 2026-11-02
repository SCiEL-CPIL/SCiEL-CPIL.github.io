"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 maximizing sum of radii.
    
    Approach: 
    1. Initialize with a reasonable grid layout.
    2. Use gradient-based optimization with penalty terms for:
       - Circles staying within [0,1]^2
       - Circles being disjoint (distance between centers >= sum of radii)
    3. Maximize sum of radii subject to constraints.
    """
    n = N
    
    # Initial configuration: use a hexagonal-ish packing or grid
    # Start with a grid and then optimize
    cols = 5
    rows = 6  # 5*6 = 30, take first 26
    # Or use 5x5=25 + 1, or just 5x6 grid
    # Let's use a 5x6 grid but only take 26
    
    # Better: start with equal radii in a grid pattern
    # For 26 circles, a 5x6 grid works (30 cells, use 26)
    # Or 6x5
    
    # Let's try a different initialization: 
    # Place circles in a roughly hexagonal pattern
    # Or just use a grid and let optimization do its work
    
    # Grid initialization
    side = int(np.ceil(np.sqrt(n)))
    # 5x6 grid
    cols, rows = 5, 6
    r0 = 0.5 / max(cols, rows) * 0.9  # initial radius
    
    centers = []
    radii = []
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n:
                break
            x = (i + 0.5) / cols
            y = (j + 0.5) / rows
            centers.append([x, y])
            radii.append(r0)
            count += 1
    
    centers = np.array(centers)
    radii = np.array(radii)
    
    # Parameters to optimize: x, y, r for each circle
    # Total 3*n parameters
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i][0]
        x0[3*i+1] = centers[i][1]
        x0[3*i+2] = radii[i]
    
    def objective(params):
        """Negative of sum of radii (we minimize negative sum) + penalties"""
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Penalty for circles outside [0,1]^2
        # r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        # Penalty: sum of max(0, r_i - min(x_i, 1-x_i, y_i, 1-y_i))^2
        boundary_violation = 0.0
        for i in range(n):
            min_dist = min(x[i], 1-x[i], y[i], 1-y[i])
            if r[i] > min_dist:
                boundary_violation += (r[i] - min_dist) ** 2
            # Also ensure r >= 0
            if r[i] < 0:
                boundary_violation += (0 - r[i]) ** 2
        
        # Penalty for overlapping circles
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2
        
        # Objective: negative sum of radii + large penalty
        sum_radii = np.sum(r)
        penalty = 100.0 * (boundary_violation + overlap_penalty)
        
        return -sum_radii + penalty
    
    def objective_grad(params):
        """Gradient of the objective function"""
        x = params[0::3].copy()
        y = params[1::3].copy()
        r = params[2::3].copy()
        
        grad = np.zeros_like(params)
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        # Gradient of boundary penalty
        for i in range(n):
            min_dist = min(x[i], 1-x[i], y[i], 1-y[i])
            # d/dx_i of max(0, r_i - min_dist)^2
            # Need to know which of x, 1-x, y, 1-y is the min
            if r[i] > min_dist:
                # Determine which constraint is active
                if x[i] <= 1-x[i] and x[i] <= y[i] and x[i] <= 1-y[i]:
                    # min is x[i], d(min_dist)/dx = 1
                    grad[3*i] += 200.0 * (r[i] - min_dist) * 1.0
                elif 1-x[i] <= x[i] and 1-x[i] <= y[i] and 1-x[i] <= 1-y[i]:
                    # min is 1-x[i], d(min_dist)/dx = -1
                    grad[3*i] += 200.0 * (r[i] - min_dist) * (-1.0)
                elif y[i] <= x[i] and y[i] <= 1-x[i] and y[i] <= 1-y[i]:
                    # min is y[i], d(min_dist)/dy = 1
                    grad[3*i+1] += 200.0 * (r[i] - min_dist) * 1.0
                else:
                    # min is 1-y[i], d(min_dist)/dy = -1
                    grad[3*i+1] += 200.0 * (r[i] - min_dist) * (-1.0)
            
            if r[i] < 0:
                grad[3*i+2] += 200.0 * (0 - r[i])
        
        # Gradient of overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    diff = min_dist - dist
                    # d/dx_i of diff^2 = 2*diff * d(diff)/dx_i
                    # d(diff)/dx_i = d(min_dist - dist)/dx_i = -d(dist)/dx_i
                    # d(dist)/dx_i = dx/dist
                    grad[3*i] += 200.0 * 2 * diff * (-dx / dist)
                    grad[3*i+1] += 200.0 * 2 * diff * (-dy / dist)
                    grad[3*i+2] += 200.0 * 2 * diff * 1.0
                    
                    grad[3*j] += 200.0 * 2 * diff * (dx / dist)
                    grad[3*j+1] += 200.0 * 2 * diff * (dy / dist)
                    grad[3*j+2] += 200.0 * 2 * diff * 1.0
        
        return grad
    
    # Optimize using L-BFGS-B
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=objective_grad,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    params = result.x
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Clip radii to be non-negative and within bounds
    r = np.maximum(r, 0)
    
    # Final cleanup: ensure circles are within [0,1]^2
    for i in range(n):
        max_r = min(x[i], 1-x[i], y[i], 1-y[i])
        if r[i] > max_r:
            r[i] = max_r * (1 - 1e-6)
    
    # Build result array
    result_arr = np.zeros((n, 3))
    for i in range(n):
        result_arr[i] = [x[i], y[i], r[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
