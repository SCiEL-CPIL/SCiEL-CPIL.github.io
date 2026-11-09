import numpy as np
import time
import random
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 axis-aligned squares in the unit square.
    Uses a penalty-based local search with multiple restarts.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 500  # Safety margin for 600s limit
    
    best_centers = None
    best_sides = None
    best_value = 0.0
    best_penalty = float('inf')
    
    # Helper to evaluate a configuration
    def evaluate(centers, sides):
        # Check bounds
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return -1e10, 1e10
        if np.any(sides <= 0):
            return -1e10, 1e10
        
        # Check containment
        # [cx - s/2, cx + s/2] in [0, 1]
        min_x = np.min(centers[:, 0] - sides / 2.0)
        max_x = np.max(centers[:, 0] + sides / 2.0)
        min_y = np.min(centers[:, 1] - sides / 2.0)
        max_y = np.max(centers[:, 1] + sides / 2.0)
        
        tol = 1e-7
        if min_x < -tol or max_x > 1 + tol or min_y < -tol or max_y > 1 + tol:
            # Return a very low score if invalid
            return -1e10, 1e10
        
        # Check overlap
        total_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                req_x = (sides[i] + sides[j]) / 2.0
                req_y = (sides[i] + sides[j]) / 2.0
                
                # Overlap amounts
                ox = max(0.0, req_x - dx)
                oy = max(0.0, req_y - dy)
                
                if ox > 0 and oy > 0:
                    # They overlap
                    total_penalty += (ox * oy) ** 2
        
        value = np.sum(sides)
        return value, total_penalty

    def objective(x):
        """
        x: shape (30,) -> [cx1, cy1, s1, cx2, cy2, s2, ...]
        Returns scalar cost to minimize.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            centers[i, 0] = x[3*i]
            centers[i, 1] = x[3*i + 1]
            sides[i] = x[3*i + 2]
            
        value, penalty = evaluate(centers, sides)
        # We want to maximize value, so minimize -value.
        # Add penalty to force validity.
        # If invalid, value is -1e10, so -value is 1e10.
        # Penalty is added to cost.
        return -value + 100.0 * penalty

    def gradient(x):
        # Numerical gradient or just use Powell (no gradient needed)
        # Powell is robust.
        pass

    # Initial configurations
    inits = []
    
    # 1. 4x4 grid subset
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    x_init = np.zeros(3 * n)
    for i in range(n):
        x_init[3*i] = cx[i]
        x_init[3*i+1] = cy[i]
        x_init[3*i+2] = side
    inits.append(x_init)
    
    # 2. 3x3 grid + 1 small (try to fit 10th in the middle or corner)
    # 9 squares of size 0.3, 1 square of size 0.1?
    # Let's just use random inits with varying scales
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Generate random init
        # Mix of grid-like and random
        if restart % 5 == 0:
            # Perturb the grid init
            x0 = inits[0] + np.random.normal(0, 0.05, 3*n)
            # Ensure sides are positive
            x0[2::3] = np.abs(x0[2::3]) + 0.1
        else:
            # Random init
            cx = np.random.uniform(0.1, 0.9, n)
            cy = np.random.uniform(0.1, 0.9, n)
            s = np.random.uniform(0.1, 0.4, n)
            x0 = np.zeros(3*n)
            for i in range(n):
                x0[3*i] = cx[i]
                x0[3*i+1] = cy[i]
                x0[3*i+2] = s[i]
        
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # cx
            bounds.append((0.0, 1.0)) # cy
            bounds.append((1e-6, 1.0)) # s
            
        # Use Powell method as it doesn't require gradients and handles non-smooth penalties well
        try:
            res = minimize(objective, x0, method='Powell', bounds=bounds, 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < 1e9:
                x_opt = res.x
                centers = np.zeros((n, 2))
                sides = np.zeros(n)
                for i in range(n):
                    centers[i, 0] = x_opt[3*i]
                    centers[i, 1] = x_opt[3*i + 1]
                    sides[i] = x_opt[3*i + 2]
                
                value, penalty = evaluate(centers, sides)
                
                # If valid (low penalty), update best
                if penalty < 1e-5:
                    # Clamp to ensure strict validity within tol
                    # The evaluator uses tol=1e-7. 
                    # If penalty is 0, they are separated.
                    # If penalty is tiny, they might be touching or slightly overlapping.
                    # We should nudge them apart if necessary, or just trust the optimizer.
                    # Let's verify validity again with strict check.
                    if np.all(centers[:, 0] - sides/2 >= -1e-9) and np.all(centers[:, 0] + sides/2 <= 1 + 1e-9) and \
                       np.all(centers[:, 1] - sides/2 >= -1e-9) and np.all(centers[:, 1] + sides/2 <= 1 + 1e-9):
                        # Check overlap strictly
                        valid = True
                        for i in range(n):
                            for j in range(i+1, n):
                                t = (sides[i] + sides[j]) / 2.0 - 1e-7
                                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                                    valid = False
                                    break
                            if not valid: break
                        
                        if valid:
                            if value > best_value:
                                best_value = value
                                best_centers = centers.copy()
                                best_sides = sides.copy()
                                
        except Exception as e:
            continue

    # Fallback: Return the baseline grid if no better found
    if best_centers is None:
        # Use the 4x4 grid subset which is always valid
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
        best_value = np.sum(best_sides)
        
    return best_centers, best_sides
