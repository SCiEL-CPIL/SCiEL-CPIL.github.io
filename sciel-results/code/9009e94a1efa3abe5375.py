# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _check_validity(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    # Check finite
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    # Check side > 0
    if np.any(sides <= 0):
        return False
    # Check inside unit square (with tol)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check no overlap
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True

def _objective(centers, sides):
    """Return sum of sides if valid, else 0."""
    if _check_validity(centers, sides):
        return np.sum(sides)
    return 0.0

def _penalized_objective(centers, sides, penalty_weight=100.0):
    """
    Smooth penalized objective for optimization.
    Returns sum(sides) - penalty for violations.
    """
    n = len(sides)
    obj = np.sum(sides)
    
    # Boundary penalty
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Distance outside boundary
        d_left = max(0, s/2 - cx)
        d_right = max(0, cx + s/2 - 1)
        d_bottom = max(0, s/2 - cy)
        d_top = max(0, cy + s/2 - 1)
        obj -= penalty_weight * (d_left**2 + d_right**2 + d_bottom**2 + d_top**2)
    
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            t = (sides[i] + sides[j])/2
            # Overlap in x: max(0, t - dx)
            # Overlap in y: max(0, t - dy)
            # Penalty for both overlapping
            ox = max(0, t - dx)
            oy = max(0, t - dy)
            # We want to penalize when both are positive
            obj -= penalty_weight * ox * oy
    
    return obj

def _local_optimize(centers, sides, max_iter=500, lr=0.01, penalty_weight=100.0):
    """
    Simple gradient-free local optimization using coordinate descent with random perturbations.
    """
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _penalized_objective(best_centers, best_sides, penalty_weight)
    
    n = len(sides)
    dim = 2 * n + n  # 2*n for centers, n for sides
    
    for it in range(max_iter):
        # Try perturbing each parameter
        improved = False
        for _ in range(10):  # 10 random perturbations per iteration
            new_centers = best_centers.copy()
            new_sides = best_sides.copy()
            
            # Random perturbation
            idx = np.random.randint(0, dim)
            if idx < 2*n:
                # Center coordinate
                comp = idx % 2
                sq_idx = idx // 2
                new_centers[sq_idx, comp] += np.random.uniform(-lr, lr)
            else:
                # Side length
                sq_idx = idx - 2*n
                new_sides[sq_idx] += np.random.uniform(-lr, lr)
                new_sides[sq_idx] = max(0.01, new_sides[sq_idx])  # Keep positive
            
            new_obj = _penalized_objective(new_centers, new_sides, penalty_weight)
            
            if new_obj > best_obj:
                best_centers = new_centers
                best_sides = new_sides
                best_obj = new_obj
                improved = True
        
        if not improved:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return best_centers, best_sides, best_obj

def _grid_packing():
    """Baseline grid packing."""
    n = 10
    k = 4  # 4x4 grid, use 10 cells
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides

def _structural_packing_05_025():
    """
    Structural packing with one large square (side 0.5) and smaller squares.
    This is a known high-performing configuration for n=10.
    """
    # One large square of side 0.5 in a corner
    # Remaining space can fit smaller squares
    
    # Let's try: one 0.5 square, and pack the rest
    # Actually, let's use a known good configuration:
    # Place squares to maximize sum
    
    # Simple approach: use a 2x2 grid of 0.5 squares (4 squares, sum=2.0)
    # Plus 6 smaller squares in the remaining space
    
    # Better: use mixed sizes
    # Let's initialize with a reasonable configuration and let optimizer work
    
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # Start with a 2x2 grid of 0.5 squares
    for i in range(4):
        row = i // 2
        col = i % 2
        centers[i, 0] = col * 0.5 + 0.25
        centers[i, 1] = row * 0.5 + 0.25
        sides[i] = 0.5
    
    # Add 6 smaller squares, try to place them
    # For now, just make them small and let optimizer move them
    for i in range(4, 10):
        centers[i, 0] = 0.5
        centers[i, 1] = 0.5
        sides[i] = 0.1
    
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin
    
    # Start with grid packing
    centers, sides = _grid_packing()
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _objective(best_centers, best_sides)
    
    # Try structural initialization
    centers2, sides2 = _structural_packing_05_025()
    obj2 = _objective(centers2, sides2)
    if obj2 > best_obj:
        best_centers = centers2.copy()
        best_sides = sides2.copy()
        best_obj = obj2
    
    # Local optimization from multiple starts
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random perturbation of best solution
        centers = best_centers + np.random.normal(0, 0.05, size=(10, 2))
        sides = np.clip(best_sides + np.random.normal(0, 0.05, size=10), 0.01, 0.6)
        
        # Local optimization
        c_opt, s_opt, obj_opt = _local_optimize(centers, sides, max_iter=200, lr=0.01)
        
        # Check validity
        if _check_validity(c_opt, s_opt):
            actual_obj = np.sum(s_opt)
            if actual_obj > best_obj:
                best_centers = c_opt.copy()
                best_sides = s_opt.copy()
                best_obj = actual_obj
    
    # Return best valid solution
    if not _check_validity(best_centers, best_sides):
        # Fallback to grid
        best_centers, best_sides = _grid_packing()
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
