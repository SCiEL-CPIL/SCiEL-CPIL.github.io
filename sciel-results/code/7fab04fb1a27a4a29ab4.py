# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    
    # We'll use a recursive guillotine cut approach to generate a valid packing,
    # then optimize the sizes and positions to maximize the sum of side lengths.
    
    # First, let's try a simple but effective approach: 
    # Use a known good configuration for 10 squares in a unit square.
    # The optimal or near-optimal packing for 10 squares is known.
    
    # One good configuration: 
    # - Place squares in a pattern that allows larger squares
    # - Use a mix of different sized squares
    
    # Let's use a heuristic approach: start with a grid-like packing and optimize
    
    # Initial guess: 4x3 grid with some squares removed, or a more optimal arrangement
    # For 10 squares, a good arrangement might be:
    # - 3 large squares in a row/column
    # - Smaller squares filling the gaps
    
    # Let's try a specific known-good configuration and then refine
    
    # Configuration 1: Based on known results for 10 squares
    # The best known packing for 10 squares has a sum of side lengths around 3.0-3.2
    
    # Let's use a deterministic initial configuration:
    # Place squares in a pattern that's known to be feasible
    
    # Start with a 3x4 grid (12 cells), use 10 of them
    k_rows = 3
    k_cols = 4
    base_side = 1.0 / max(k_rows, k_cols)
    
    # Actually, let's use a more sophisticated approach:
    # Start with all squares equal, then optimize
    
    # Initial configuration: 10 squares in a 4x3 grid pattern
    # Cells: (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2)
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 4 columns, 3 rows, use first 10 cells
    for i in range(n):
        col = i % 4
        row = i // 4
        cx = (col + 0.5) / 4.0
        cy = (row + 0.5) / 3.0
        centers[i] = [cx, cy]
        sides[i] = 1.0 / 4.0
    
    # Now optimize: we want to maximize sum(sides) subject to constraints:
    # 1. Each square inside [0,1]^2
    # 2. No overlap between any pair of squares
    
    def objective(params):
        """Negative sum of sides (we minimize negative to maximize sum)"""
        # params: 20 values: 10 centers (x,y) + 10 sides
        c = params[:20].reshape(10, 2)
        s = params[20:]
        
        # Penalty for going outside [0,1]
        penalty = 0.0
        for i in range(10):
            x, y = c[i]
            side = s[i]
            # Square must be in [0,1]^2
            if x - side/2 < 0:
                penalty += (x - side/2)**2
            if x + side/2 > 1:
                penalty += (x + side/2 - 1)**2
            if y - side/2 < 0:
                penalty += (y - side/2)**2
            if y + side/2 > 1:
                penalty += (y + side/2 - 1)**2
        
        # Penalty for overlap
        for i in range(10):
            for j in range(i+1, 10):
                xi, yi = c[i]
                xj, yj = c[j]
                si = s[i]
                sj = s[j]
                
                # Check if squares overlap
                # Overlap if |xi-xj| < (si+sj)/2 AND |yi-yj| < (si+sj)/2
                dx = abs(xi - xj)
                dy = abs(yi - yj)
                thresh = (si + sj) / 2.0
                
                if dx < thresh and dy < thresh:
                    # Overlap - penalize
                    overlap_x = thresh - dx
                    overlap_y = thresh - dy
                    penalty += 100.0 * (overlap_x**2 + overlap_y**2)
        
        return -np.sum(s) + penalty
    
    # Set up optimization
    # Parameters: 10 centers (2 each) + 10 sides = 30 parameters
    x0 = np.concatenate([centers.flatten(), sides])
    
    # Bounds: centers in [0,1], sides in (0, 1]
    bounds = []
    for i in range(10):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
    for i in range(10):
        bounds.append((0.01, 1.0))  # side
    
    # Try multiple random starts
    best_params = x0
    best_obj = objective(x0)
    
    np.random.seed(42)
    for trial in range(50):
        # Perturb initial guess
        if trial == 0:
            x_trial = x0.copy()
        else:
            x_trial = x0 + np.random.randn(30) * 0.1
            # Clip to valid range
            for i in range(10):
                x_trial[2*i] = np.clip(x_trial[2*i], 0.0, 1.0)
                x_trial[2*i+1] = np.clip(x_trial[2*i+1], 0.0, 1.0)
            for i in range(10):
                x_trial[20+i] = np.clip(x_trial[20+i], 0.01, 1.0)
        
        try:
            result = minimize(objective, x_trial, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000, 'ftol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except:
            pass
    
    # Extract best solution
    centers_opt = best_params[:20].reshape(10, 2)
    sides_opt = best_params[20:]
    
    # Verify the solution is valid
    tol = 1e-7
    valid = True
    for i in range(10):
        cx, cy = centers_opt[i]
        s = sides_opt[i]
        if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(s):
            valid = False
            break
        if s <= 0:
            valid = False
            break
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            valid = False
            break
    
    if valid:
        for i in range(10):
            for j in range(i+1, 10):
                xi, yi = centers_opt[i]
                xj, yj = centers_opt[j]
                si = sides_opt[i]
                sj = sides_opt[j]
                t = (si + sj) / 2.0 - tol
                if abs(xi - xj) < t and abs(yi - yj) < t:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback to the initial grid configuration
        return centers, sides
    
    return centers_opt, sides_opt


# EVOLVE-BLOCK-END
