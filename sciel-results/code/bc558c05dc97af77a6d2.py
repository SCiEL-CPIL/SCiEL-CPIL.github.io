# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validate(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check bounds
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
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True

def _objective(x, n, penalty_weight=1000.0):
    """Objective: minimize negative sum of sides + penalty for violations."""
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Objective: maximize sum of sides
    obj = -np.sum(sides)
    
    # Penalty for going outside [0,1]^2
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            obj += penalty_weight * 10
        else:
            # Left boundary violation
            if cx - s/2 < 0:
                obj += penalty_weight * (cx - s/2) ** 2
            # Right boundary violation
            if cx + s/2 > 1:
                obj += penalty_weight * (cx + s/2 - 1) ** 2
            # Bottom boundary violation
            if cy - s/2 < 0:
                obj += penalty_weight * (cy - s/2) ** 2
            # Top boundary violation
            if cy + s/2 > 1:
                obj += penalty_weight * (cy + s/2 - 1) ** 2
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            # If both dx < t and dy < t, they overlap
            # Use smooth penalty
            if dx < t and dy < t:
                # How much they overlap in both dimensions
                overlap_x = t - dx
                overlap_y = t - dy
                obj += penalty_weight * (overlap_x ** 2 + overlap_y ** 2)
    
    # Penalty for non-positive sides
    for i in range(n):
        if sides[i] <= 0:
            obj += penalty_weight * sides[i] ** 2
    
    return obj

def _initial_grid(n=10):
    """Create initial grid packing."""
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides

def _local_search(centers, sides, n=10, max_iter=500, tol=1e-10):
    """Local search using coordinate descent with step halving."""
    centers = centers.copy()
    sides = sides.copy()
    best_sum = np.sum(sides)
    
    for iteration in range(max_iter):
        improved = False
        
        # Try to adjust each square's position and size
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            
            # Try increasing side length
            for delta_s in [0.01, 0.005, 0.001, 0.0001]:
                new_s = s + delta_s
                if new_s <= 0:
                    continue
                # Check if we can fit
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    t = (new_s + sides[j]) / 2
                    if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                        valid = False
                        break
                if not valid:
                    continue
                # Check bounds
                if cx - new_s/2 < -tol or cx + new_s/2 > 1 + tol:
                    continue
                if cy - new_s/2 < -tol or cy + new_s/2 > 1 + tol:
                    continue
                # Accept
                sides[i] = new_s
                improved = True
                break
            
            # Try moving in each direction
            for dx, dy in [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01),
                          (0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005),
                          (0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001)]:
                new_cx = cx + dx
                new_cy = cy + dy
                # Check bounds
                if new_cx - s/2 < -tol or new_cx + s/2 > 1 + tol:
                    continue
                if new_cy - s/2 < -tol or new_cy + s/2 > 1 + tol:
                    continue
                # Check overlaps
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    t = (s + sides[j]) / 2
                    if abs(new_cx - centers[j,0]) < t and abs(new_cy - centers[j,1]) < t:
                        valid = False
                        break
                if not valid:
                    continue
                centers[i] = [new_cx, new_cy]
                improved = True
                break
        
        if not improved:
            break
        
        current_sum = np.sum(sides)
        if current_sum > best_sum:
            best_sum = current_sum
    
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square to maximize sum of side lengths.
    
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides: np.ndarray of shape (10,)
    """
    n = 10
    start_time = time.time()
    deadline = 580  # Safety margin before 630s limit
    
    # Start with grid packing
    centers, sides = _initial_grid(n)
    best_centers, best_sides = centers.copy(), sides.copy()
    best_sum = np.sum(sides)
    
    # Try multiple initial configurations
    configs = [
        _initial_grid(n),
    ]
    
    # Add some perturbed initial configs
    np.random.seed(42)
    for _ in range(5):
        c, s = _initial_grid(n)
        c += np.random.randn(n, 2) * 0.01
        s += np.random.randn(n) * 0.01
        s = np.clip(s, 0.05, 0.5)
        configs.append((c, s))
    
    for centers_init, sides_init in configs:
        if time.time() > deadline:
            break
        
        # Local search
        c_opt, s_opt = _local_search(centers_init, sides_init, n, max_iter=200)
        
        if _validate(c_opt, s_opt):
            s_sum = np.sum(s_opt)
            if s_sum > best_sum:
                best_sum = s_sum
                best_centers, best_sides = c_opt.copy(), s_opt.copy()
    
    # Final polish: try to squeeze out more with smaller steps
    if time.time() < deadline - 30:
        c_polish, s_polish = _local_search(best_centers, best_sides, n, max_iter=1000, tol=1e-12)
        if _validate(c_polish, s_polish):
            s_sum = np.sum(s_polish)
            if s_sum > best_sum:
                best_sum = s_sum
                best_centers, best_sides = c_polish.copy(), s_polish.copy()
    
    # Ensure validity
    if not _validate(best_centers, best_sides):
        # Fall back to grid
        best_centers, best_sides = _initial_grid(n)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
