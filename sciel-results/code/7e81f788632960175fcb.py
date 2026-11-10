import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 590.0  # Safety margin for 600s limit
    
    def get_sides_iterative(centers, max_iters=50):
        """
        Computes the maximum side lengths for squares centered at 'centers'
        using iterative relaxation.
        """
        c = centers.reshape(n, 2)
        # Initialize with boundary limits
        sides = np.zeros(n)
        for i in range(n):
            ci = c[i]
            sides[i] = min(2.0 * ci[0], 2.0 * (1.0 - ci[0]), 2.0 * ci[1], 2.0 * (1.0 - ci[1]))
            if sides[i] < 1e-6:
                sides[i] = 1e-6
                
        # Iterative Gauss-Seidel update
        for _ in range(max_iters):
            changed = False
            for i in range(n):
                ci = c[i]
                max_s = min(2.0 * ci[0], 2.0 * (1.0 - ci[0]), 2.0 * ci[1], 2.0 * (1.0 - ci[1]))
                if max_s < 1e-6:
                    max_s = 1e-6
                
                for j in range(n):
                    if i == j:
                        continue
                    cj = c[j]
                    dx = abs(ci[0] - cj[0])
                    dy = abs(ci[1] - cj[1])
                    sj = sides[j]
                    
                    # Constraint: |dx| >= (s_i + s_j)/2 OR |dy| >= (s_i + s_j)/2
                    # s_i <= 2|dx| - s_j  OR  s_i <= 2|dy| - s_j
                    # s_i <= max(2|dx| - s_j, 2|dy| - s_j)
                    limit_x = 2.0 * dx - sj
                    limit_y = 2.0 * dy - sj
                    limit = max(limit_x, limit_y)
                    
                    if limit < max_s:
                        max_s = limit
                
                if max_s < 1e-6:
                    max_s = 1e-6
                
                if abs(max_s - sides[i]) > 1e-9:
                    sides[i] = max_s
                    changed = True
            if not changed:
                break
        return sides

    def is_valid(centers, sides, tol=1e-7):
        """Check validity constraints."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Boundary check
        if np.any(centers - sides[:, None] / 2.0 < -tol):
            return False
        if np.any(centers + sides[:, None] / 2.0 > 1.0 + tol):
            return False
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    return False
        return True

    def clamp_centers(centers, sides):
        """Clamp centers so squares stay inside [0,1]^2."""
        c = centers.copy()
        for i in range(n):
            half = sides[i] / 2.0
            c[i, 0] = max(half, min(1.0 - half, c[i, 0]))
            c[i, 1] = max(half, min(1.0 - half, c[i, 1]))
        return c

    def objective(params):
        c = params.reshape(n, 2)
        sides = get_sides_iterative(c, max_iters=10) # Fewer iters for speed during optimization
        
        # Penalty for invalid states (though sides should be valid by construction, 
        # numerical issues or extreme positions might cause issues)
        if not is_valid(c, sides, tol=1e-5):
            return 1e6 - np.sum(sides)
            
        return -np.sum(sides)

    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    np.random.seed(42)
    num_starts = 100 # Will be limited by time
    
    for k in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization, slightly biased towards center to avoid corners initially?
        # Or just uniform.
        centers_init = np.random.rand(n, 2)
        
        # Optimize
        try:
            res = minimize(
                objective,
                centers_init.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * n),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            centers_opt = res.x.reshape(n, 2)
            
            # Post-processing: Clamp and grow
            centers_opt = clamp_centers(centers_opt, get_sides_iterative(centers_opt, max_iters=10))
            sides_opt = get_sides_iterative(centers_opt, max_iters=50)
            
            # Final validity check
            if is_valid(centers_opt, sides_opt):
                score = np.sum(sides_opt)
                if score > best_score:
                    best_score = score
                    best_centers = centers_opt
                    best_sides = sides_opt
        except Exception:
            continue

    # Fallback if no valid solution found
    if best_centers is None:
        # Simple grid fallback
        k_grid = 4
        side = 1.0 / k_grid
        idx = np.arange(n)
        cols = idx % k_grid
        rows = idx // k_grid
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
        
        # Verify fallback
        if not is_valid(best_centers, best_sides):
             # Should not happen for 4x4 grid with 10 items
             pass

    return best_centers, best_sides
