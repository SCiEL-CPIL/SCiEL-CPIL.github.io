"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 to maximize sum of radii.
    
    Approach:
    1. Parametrize the rectangle by aspect ratio R = w/h.
    2. For each R, use SLSQP to optimize circle positions and radii with explicit constraints.
    3. Search over R to find the global optimum.
    """
    
    def solve_for_aspect_ratio(R, seed_val):
        """
        Solve the packing problem for a fixed aspect ratio R.
        Returns the sum of radii and the optimal parameters.
        """
        random.seed(seed_val)
        
        # Calculate w and h from R and perimeter constraint 2*(w+h)=4 => w+h=2
        # w = R*h => h(R+1) = 2 => h = 2/(R+1), w = 2R/(R+1)
        h = 2.0 / (R + 1.0)
        w = R * h
        
        # Initial guess: Grid
        side = int(np.ceil(np.sqrt(N)))
        init = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side * w
            y = (row + 0.5) / side * h
            r = min(w, h) / (2 * side) * 0.9
            init.extend([x, y, r])
        
        x0 = np.array(init)
        
        # Objective: Minimize negative sum of radii
        def objective(params):
            radii = params[2::3]
            return -np.sum(radii)
        
        # Constraints:
        # 1. Boundary constraints: x_i >= r_i, x_i <= w - r_i, etc.
        #    We can express these as: x_i - r_i >= 0, w - x_i - r_i >= 0, etc.
        # 2. Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        def boundary_constraints(params):
            cons = []
            circles = params.reshape((-1, 3))
            for i in range(N):
                x, y, r = circles[i]
                # x >= r
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
                # w - x >= r => w - x - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: w - p[3*i] - p[3*i+2]})
                # y >= r
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
                # h - y >= r => h - y - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: h - p[3*i+1] - p[3*i+2]})
            return cons
        
        def non_overlap_constraints(params):
            cons = []
            circles = params.reshape((-1, 3))
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
                    })
            return cons
        
        constraints = boundary_constraints(x0) + non_overlap_constraints(x0)
        
        # Bounds: radii must be positive
        bounds = []
        for i in range(N):
            bounds.append((0, w))  # x
            bounds.append((0, h))  # y
            bounds.append((1e-6, min(w, h)/2))  # r
            
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                              options={'maxiter': 500, 'ftol': 1e-10})
            
            if result.success:
                return -result.fun, result.x
            else:
                # If it fails, try to recover by clamping and checking validity
                # For now, return a low value
                return -np.inf, x0
                
        except Exception as e:
            return -np.inf, x0

    # Search over aspect ratios
    best_sum_radii = 0
    best_params = None
    best_R = 1.0
    
    # Try a range of aspect ratios
    # The optimal aspect ratio for packing circles in a rectangle is often close to 1 or specific ratios.
    # Let's try R from 0.5 to 2.0
    R_values = np.linspace(0.5, 2.0, 15)
    
    for R in R_values:
        # Use a few random restarts for each R to avoid local minima
        best_for_R = -np.inf
        best_params_for_R = None
        
        for restart in range(3):
            # Perturb initial guess slightly
            sum_r, params = solve_for_aspect_ratio(R, 42 + restart)
            if sum_r > best_for_R:
                best_for_R = sum_r
                best_params_for_R = params
        
        if best_for_R > best_sum_radii:
            best_sum_radii = best_for_R
            best_params = best_params_for_R
            best_R = R
            
    # Reconstruct the final answer from best_params
    if best_params is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    circles = best_params.reshape((-1, 3))
    
    # Final validation and cleanup
    # Ensure perimeter constraint is satisfied (it is by construction of w,h)
    # Ensure circles are within bounds and non-overlapping
    # SLSQP should have handled this, but let's do a quick check and clamp if necessary
    h = 2.0 / (best_R + 1.0)
    w = best_R * h
    
    for i in range(N):
        cx, cy, cr = circles[i]
        # Clamp to bounds
        cr = max(cr, 1e-6)
        cx = min(max(cx, cr), w - cr)
        cy = min(max(cy, cr), h - cr)
        circles[i] = [cx, cy, cr]
        
    # Check for overlaps and shrink if necessary (safety net)
    for _ in range(10):
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    excess = min_dist - dist
                    # Shrink both circles
                    factor = 1.0 - 0.01 * excess
                    ri *= max(factor, 0.5)
                    rj *= max(factor, 0.5)
                    circles[i, 2] = ri
                    circles[j, 2] = rj
                    valid = False
        if valid:
            break
            
    return circles
# EVOLVE-BLOCK-END
