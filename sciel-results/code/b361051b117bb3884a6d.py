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
    best_val = -np.inf
    best_centers = None
    best_sides = None

    def compute_penalty(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        sides = np.maximum(sides, 1e-10)
        
        penalty = 0.0
        
        # Boundary penalties
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Distance to boundaries
            d_left = cx - s/2
            d_right = 1.0 - (cx + s/2)
            d_bottom = cy - s/2
            d_top = 1.0 - (cy + s/2)
            
            # Penalize if outside [-tol, 1+tol]
            if d_left < -tol:
                penalty += (d_left + tol)**2
            if d_right < -tol:
                penalty += (d_right + tol)**2
            if d_bottom < -tol:
                penalty += (d_bottom + tol)**2
            if d_top < -tol:
                penalty += (d_top + tol)**2

        # Overlap penalties
        # We use a sum of squares of the "violation depth" in each dimension
        # This ensures gradients exist even if only one dimension is violated
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i][0] - centers[j][0])
                dy = abs(centers[i][1] - centers[j][1])
                t = (sides[i] + sides[j])/2 - tol
                
                # If t <= 0, they cannot overlap (sides too small or tol large)
                if t > 0:
                    px = max(0.0, t - dx)
                    py = max(0.0, t - dy)
                    # Sum of squares provides better gradient signal than product
                    penalty += px**2 + py**2
        
        return penalty

    def objective(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        sides = np.maximum(sides, 1e-10)
        
        obj = np.sum(sides)
        pen = compute_penalty(x)
        
        # Large weight for penalty to ensure validity
        return -(obj - 1000.0 * pen)

    def is_valid(centers, sides):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
            
        # Check boundaries
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
                
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i][0] - centers[j][0])
                dy = abs(centers[i][1] - centers[j][1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    return False
        return True

    def get_value(centers, sides):
        if not is_valid(centers, sides):
            return -np.inf
        return np.sum(sides)

    # Multi-start optimization
    # Number of random restarts
    n_restarts = 50
    rng = np.random.default_rng(42)
    
    # Bounds for variables
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # cx
        bounds.append((0, 1)) # cy
        bounds.append((1e-4, 1.0)) # s (small min to avoid 0 issues)

    for r in range(n_restarts):
        # Random initialization
        # Centers in [0.1, 0.9] to allow some growth
        centers_init = rng.uniform(0.1, 0.9, (n, 2))
        # Sides start small, e.g., 0.1
        sides_init = rng.uniform(0.1, 0.2, n)
        
        x0 = np.concatenate([centers_init.flatten(), sides_init])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            x_opt = result.x
            centers = x_opt[:2*n].reshape(n, 2)
            sides = x_opt[2*n:]
            
            # Ensure sides are positive
            sides = np.maximum(sides, 1e-10)
            
            val = get_value(centers, sides)
            
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_sides = sides.copy()
                
        except Exception:
            continue
            
    # Fallback to grid if no valid solution found (unlikely)
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)

# EVOLVE-BLOCK-END
