# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize the sum of side lengths.
    Uses a multi-start local optimization approach with penalty-based constraints.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Initialize with a good grid packing as the base
    k = int(np.ceil(np.sqrt(n)))
    side0 = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side0
    cy0 = (rows + 0.5) * side0
    
    best_centers = np.stack([cx0, cy0], axis=1).astype(np.float64)
    best_sides = np.full(n, side0, dtype=np.float64)
    best_score = np.sum(best_sides)
    
    def compute_overlap_penalty(centers, sides):
        """Compute penalty for overlaps and boundary violations."""
        cx, cy = centers[:, 0], centers[:, 1]
        half_s = sides / 2.0
        
        # Boundary penalty
        boundary_violation = 0.0
        for i in range(n):
            if cx[i] - half_s[i] < -tol:
                boundary_violation += (-tol - (cx[i] - half_s[i]))**2
            if cx[i] + half_s[i] > 1 + tol:
                boundary_violation += ((cx[i] + half_s[i]) - (1 + tol))**2
            if cy[i] - half_s[i] < -tol:
                boundary_violation += (-tol - (cy[i] - half_s[i]))**2
            if cy[i] + half_s[i] > 1 + tol:
                boundary_violation += ((cy[i] + half_s[i]) - (1 + tol))**2
        
        # Overlap penalty
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    # Penalize the overlap
                    overlap_penalty += (t - dx)**2 + (t - dy)**2
        
        return boundary_violation + overlap_penalty
    
    def is_valid(centers, sides):
        """Check if the configuration is valid."""
        cx, cy = centers[:, 0], centers[:, 1]
        half_s = sides / 2.0
        
        # Check boundaries
        for i in range(n):
            if cx[i] - half_s[i] < -tol or cx[i] + half_s[i] > 1 + tol:
                return False
            if cy[i] - half_s[i] < -tol or cy[i] + half_s[i] > 1 + tol:
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    return False
        
        # Check sides > 0
        if np.any(sides <= 0):
            return False
        
        # Check finite
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        
        return True
    
    def objective(params):
        """Objective function: minimize negative sum of sides + penalty."""
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]
        
        # Ensure sides are positive
        sides = np.maximum(sides, 0.01)
        
        penalty = compute_overlap_penalty(centers, sides)
        # Weight penalty heavily
        return -np.sum(sides) + 100.0 * penalty
    
    # Multi-start optimization
    random.seed(42)
    np.random.seed(42)
    
    num_starts = 50
    max_time_per_start = 10.0
    
    for start_iter in range(num_starts):
        if time.time() > deadline - 60:
            break
        
        # Generate initial guess
        if start_iter == 0:
            # Start with the grid packing
            init_params = np.concatenate([best_centers.flatten(), best_sides])
        else:
            # Perturb the best solution or create a random valid-ish starting point
            if start_iter % 3 == 1:
                # Perturb current best
                noise = 0.05 * np.random.randn(2*n + n)
                init_params = best_centers.flatten() + noise[:2*n]
                init_params[2*n:] = best_sides + noise[2*n:]
                init_params[2*n:] = np.maximum(init_params[2*n:], 0.1)
            else:
                # Try different grid arrangements or corner-based packings
                # Corner-based: place large squares in corners
                cx = np.random.uniform(0.1, 0.9, n)
                cy = np.random.uniform(0.1, 0.9, n)
                sides = np.random.uniform(0.15, 0.4, n)
                init_params = np.concatenate([np.stack([cx, cy], axis=1).flatten(), sides])
        
        # Run optimization
        try:
            result = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.success or result.fun < objective(init_params):
                opt_params = result.x
                opt_centers = opt_params[:2*n].reshape(n, 2)
                opt_sides = np.maximum(opt_params[2*n:], 0.01)
                
                if is_valid(opt_centers, opt_sides):
                    opt_score = np.sum(opt_sides)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_centers = opt_centers.copy()
                        best_sides = opt_sides.copy()
        except Exception:
            pass
    
    # Final polish: try to increase sides slightly while maintaining validity
    if time.time() < deadline - 30:
        # Try to expand each square slightly
        for _ in range(20):
            improved = False
            for i in range(n):
                for delta in [0.001, 0.0005, 0.0001]:
                    test_sides = best_sides.copy()
                    test_sides[i] += delta
                    if is_valid(best_centers, test_sides):
                        best_sides = test_sides
                        improved = True
                        break
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
