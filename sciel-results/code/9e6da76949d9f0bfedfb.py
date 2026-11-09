# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def squares_in_square10() -> tuple:
    """
    Place 10 squares in the unit square to maximize the sum of their side lengths.
    
    Strategy:
    Use a multi-start local optimization approach.
    1. Generate multiple random initial configurations of 10 squares.
    2. For each configuration, use scipy.optimize.minimize (Nelder-Mead) to 
       maximize the sum of side lengths while penalizing overlaps and boundary violations.
    3. Return the best valid configuration found.
    
    This explores the solution space more broadly than a fixed heuristic.
    """
    n = 10
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    def penalty(x):
        """
        Calculate penalty for a configuration.
        x is a vector of length 30: [cx0, cy0, s0, cx1, cy1, s1, ...]
        """
        centers = x.reshape(n, 3)[:, :2]
        sides = x.reshape(n, 3)[:, 2]
        
        # Boundary penalty
        pen = 0.0
        for i in range(n):
            cx, cy, s = centers[i][0], centers[i][1], sides[i]
            if cx - s/2 < 0: pen += (cx - s/2)**2
            if cx + s/2 > 1: pen += (cx + s/2 - 1)**2
            if cy - s/2 < 0: pen += (cy - s/2)**2
            if cy + s/2 > 1: pen += (cy + s/2 - 1)**2
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = (sides[i] + sides[j]) / 2.0 - abs(centers[i][0] - centers[j][0])
                dy = (sides[i] + sides[j]) / 2.0 - abs(centers[i][1] - centers[j][1])
                if dx > 0 and dy > 0:
                    # Penalize overlap area
                    pen += dx * dy * 10.0
        
        return pen

    def objective(x):
        """
        Objective function to minimize: -sum(sides) + penalty
        """
        sides = x.reshape(n, 3)[:, 2]
        # Ensure sides are positive in the objective calculation for stability
        # but the penalty will drive them to be valid
        s_sum = np.sum(np.maximum(sides, 0))
        pen = penalty(x)
        return -s_sum + pen

    def is_valid(centers, sides, tol=1e-7):
        """
        Check if the configuration is valid.
        """
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
            
        for i in range(n):
            cx, cy, s = centers[i][0], centers[i][1], sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                    return False
        return True

    # Multi-start optimization
    num_restarts = 50
    random.seed(42) # For reproducibility
    
    for _ in range(num_restarts):
        # Initialize random configuration
        # Sides between 0.1 and 0.5
        x0 = np.random.uniform(0.1, 0.5, size=3*n)
        # Centers within [0, 1]
        # We need to map x0 to centers and sides properly
        # x0 structure: [cx0, cy0, s0, ...]
        # Let's just generate random centers and sides
        centers_init = np.random.uniform(0, 1, size=(n, 2))
        sides_init = np.random.uniform(0.1, 0.4, size=n)
        
        x0 = np.zeros(3*n)
        for i in range(n):
            x0[3*i] = centers_init[i][0]
            x0[3*i+1] = centers_init[i][1]
            x0[3*i+2] = sides_init[i]
            
        # Optimize
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            x_opt = res.x
            centers_opt = x_opt.reshape(n, 3)[:, :2]
            sides_opt = x_opt.reshape(n, 3)[:, 2]
            
            # Ensure sides are positive
            sides_opt = np.maximum(sides_opt, 1e-9)
            
            # Check validity
            if is_valid(centers_opt, sides_opt):
                score = np.sum(sides_opt)
                if score > best_score:
                    best_score = score
                    best_centers = centers_opt.copy()
                    best_sides = sides_opt.copy()
        except Exception as e:
            continue

    # If no valid solution found from optimization, fallback to a simple heuristic
    if best_centers is None:
        s = 0.3333333
        L = 1.0 - 3.0 * s
        best_centers = np.zeros((n, 2), dtype=np.float64)
        best_sides = np.zeros(n, dtype=np.float64)
        for i in range(9):
            row = i // 3
            col = i % 3
            best_centers[i] = [(col + 0.5) * s, (row + 0.5) * s]
            best_sides[i] = s
        best_centers[9] = [1.0 - L / 2.0, 1.0 - L / 2.0]
        best_sides[9] = L

    return best_centers, best_sides

# EVOLVE-BLOCK-END
