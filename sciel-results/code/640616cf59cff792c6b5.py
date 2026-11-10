# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


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
    tol = 1e-7

    def compute_sides(centers):
        """
        For each square i, compute the maximum side length such that it doesn't
        overlap with any other square and stays within the unit square.
        """
        sides = np.zeros(n)
        for i in range(n):
            cx, cy = centers[i]
            # Distance to boundaries
            d_left = cx
            d_right = 1.0 - cx
            d_bottom = cy
            d_top = 1.0 - cy
            max_s_boundary = 2.0 * min(d_left, d_right, d_bottom, d_top)
            
            max_s = max_s_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = abs(cx - centers[j][0])
                dy = abs(cy - centers[j][1])
                # If squares are separated in x or y, the constraint is:
                # s_i/2 + s_j/2 <= dx (if separated in x) OR s_i/2 + s_j/2 <= dy (if separated in y)
                # But we don't know s_j yet. This is a coupled system.
                # For a simpler approach in this function, let's use an iterative or fixed-point method.
                # Actually, for the optimizer, we need a differentiable or at least smooth objective.
                # Let's use a different strategy: define the objective as sum of sides minus a penalty for violations.
            sides[i] = max_s
        return sides

    # A better approach: use a penalty-based optimization where we optimize positions and sides together.
    # Variables: x = [cx_1, cy_1, s_1, cx_2, cy_2, s_2, ..., cx_10, cy_10, s_10]
    # Objective: maximize sum(s_i) - penalty(violations)
    # Violations:
    #   1. Boundary violations: cx_i - s_i/2 < -tol or cx_i + s_i/2 > 1+tol, etc.
    #   2. Overlap violations: |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol

    def penalty_and_objective(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Ensure sides are positive
        sides = np.maximum(sides, 1e-10)
        
        # Objective: sum of sides
        obj = np.sum(sides)
        
        # Penalty for boundary violations
        penalty = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Left boundary
            if cx - s/2 < -tol:
                penalty += (cx - s/2 + tol)**2
            # Right boundary
            if cx + s/2 > 1 + tol:
                penalty += (cx + s/2 - 1 - tol)**2
            # Bottom boundary
            if cy - s/2 < -tol:
                penalty += (cy - s/2 + tol)**2
            # Top boundary
            if cy + s/2 > 1 + tol:
                penalty += (cy + s/2 - 1 - tol)**2
        
        # Penalty for overlap violations
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i][0] - centers[j][0])
                dy = abs(centers[i][1] - centers[j][1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    # Overlap penalty: proportional to how much they overlap
                    # We want to penalize when both dx < t and dy < t
                    # Use a smooth approximation: penalty = max(0, t - dx) * max(0, t - dy)
                    px = max(0, t - dx)
                    py = max(0, t - dy)
                    penalty += px * py
        
        return -(obj - 1000.0 * penalty)
    
    # Initial guess: grid packing
    k = int(np.ceil(np.sqrt(n)))
    side_init = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_init = (cols + 0.5) * side_init
    cy_init = (rows + 0.5) * side_init
    centers_init = np.stack([cx_init, cy_init], axis=1).flatten()
    sides_init = np.full(n, side_init)
    x0 = np.concatenate([centers_init, sides_init])
    
    # Bounds
    bounds = []
    for i in range(n):
        # cx_i, cy_i in [0, 1]
        bounds.append((0, 1))
        bounds.append((0, 1))
        # s_i in [0, 1]
        bounds.append((0, 1))
    
    # Optimize
    result = minimize(penalty_and_objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    x_opt = result.x
    centers = x_opt[:2*n].reshape(n, 2)
    sides = x_opt[2*n:]
    sides = np.maximum(sides, 1e-10)
    
    # Verify and fix any remaining issues
    # Ensure all values are finite
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        # Fallback to grid
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    
    return centers.astype(np.float64), sides.astype(np.float64)


# EVOLVE-BLOCK-END
