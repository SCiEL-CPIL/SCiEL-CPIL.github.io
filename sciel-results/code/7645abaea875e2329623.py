# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _validity(centers, sides, tol=1e-7):
    """Check if the packing is valid: all squares inside unit square and non-overlapping."""
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(s):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True

def _objective(x, n=10):
    """Objective function for optimization: maximize sum of sides subject to constraints.
    We minimize negative sum of sides."""
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Bounds: sides must be positive
    if np.any(sides <= 0):
        return 1e6
    
    # Check if all squares are inside the unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -1e-9 or cx + s/2 > 1 + 1e-9:
            return 1e6
        if cy - s/2 < -1e-9 or cy + s/2 > 1 + 1e-9:
            return 1e6
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            required = (sides[i] + sides[j])/2
            # If both dx and dy are less than required, there's overlap
            if dx < required and dy < required:
                # Penalty is proportional to the overlap
                penalty += (required - dx) + (required - dy)
    
    return -np.sum(sides) + penalty * 1000

def _try_optimization(initial_centers, initial_sides, max_iters=10000, tol=1e-7):
    """Try to optimize the packing starting from initial configuration."""
    n = len(initial_sides)
    x0 = np.concatenate([initial_centers.flatten(), initial_sides])
    
    # Bounds
    bounds = []
    for i in range(n):
        # Center x: [s/2, 1-s/2], but s varies, so use loose bounds
        bounds.append((0, 1))  # cx
        bounds.append((0, 1))  # cy
    for i in range(n):
        bounds.append((1e-6, 1.0))  # side
    
    result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if result.success or result.fun < 1e5:
        centers = result.x[:2*n].reshape(n, 2)
        sides = result.x[2*n:]
        # Clamp to valid range
        for i in range(n):
            s = sides[i]
            centers[i,0] = np.clip(centers[i,0], s/2, 1-s/2)
            centers[i,1] = np.clip(centers[i,1], s/2, 1-s/2)
            sides[i] = np.clip(sides[i], 1e-6, 1.0)
        
        if _validity(centers, sides, tol):
            return centers, sides, np.sum(sides)
    
    return None

def _grid_packing(n=10):
    """Simple grid packing as baseline."""
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

def _two_row_packing():
    """Packing with two rows of 5 squares each, but with different sizes."""
    # Try a configuration where we have larger squares in some positions
    # Row 1: 5 squares, Row 2: 5 squares
    # Let's try making the first row have slightly larger squares
    
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Row 1 (bottom): 5 squares
    s1 = 0.2
    for i in range(5):
        centers[i, 0] = (i + 0.5) * s1
        centers[i, 1] = s1/2
        sides[i] = s1
    
    # Row 2 (top): 5 squares
    s2 = 0.2
    for i in range(5):
        centers[i+5, 0] = (i + 0.5) * s2
        centers[i+5, 1] = 1 - s2/2
        sides[i+5] = s2
    
    return centers, sides

def _asymmetric_packing():
    """Try an asymmetric packing with different square sizes."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Place 4 large squares in a 2x2 grid in the bottom-left
    s_large = 0.25
    positions_large = [(0.125, 0.125), (0.375, 0.125), (0.125, 0.375), (0.375, 0.375)]
    for i, (cx, cy) in enumerate(positions_large):
        centers[i] = [cx, cy]
        sides[i] = s_large
    
    # Place 6 smaller squares in the remaining space
    # Top-right area: x from 0.5 to 1, y from 0 to 1
    # Left area: x from 0 to 0.5, y from 0.5 to 1
    
    # Top-right: 3x2 grid of smaller squares
    s_small1 = 1/6  # ~0.1667
    for i in range(3):
        for j in range(2):
            idx = 4 + i*2 + j
            cx = 0.5 + (i + 0.5) * s_small1
            cy = (j + 0.5) * s_small1
            centers[idx] = [cx, cy]
            sides[idx] = s_small1
    
    # Left-top: 3 squares
    s_small2 = 0.5/3  # ~0.1667
    for i in range(3):
        idx = 4 + i*2  # This doesn't work, let me redo
    
    # Let me redo this more carefully
    # Bottom-left 2x2: 4 squares of side 0.25
    # Top-right 3x2: 6 squares of side 1/6
    # That's 10 squares total
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 4 large squares in bottom-left
    s_large = 0.25
    for i in range(2):
        for j in range(2):
            idx = i*2 + j
            cx = (i + 0.5) * s_large
            cy = (j + 0.5) * s_large
            centers[idx] = [cx, cy]
            sides[idx] = s_large
    
    # 6 small squares in top-right
    s_small = 1/6
    for i in range(3):
        for j in range(2):
            idx = 4 + i*2 + j
            cx = 0.5 + (i + 0.5) * s_small
            cy = 0.5 + (j + 0.5) * s_small
            centers[idx] = [cx, cy]
            sides[idx] = s_small
    
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize the sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    best_centers, best_sides, best_sum = None, None, 0
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Grid packing
    c, s = _grid_packing()
    initial_configs.append((c, s))
    
    # 2. Two-row packing
    c, s = _two_row_packing()
    initial_configs.append((c, s))
    
    # 3. Asymmetric packing
    c, s = _asymmetric_packing()
    initial_configs.append((c, s))
    
    # 4. Random configurations
    np.random.seed(42)
    for _ in range(5):
        n = 10
        sides = np.random.uniform(0.1, 0.3, n)
        centers = np.zeros((n, 2))
        for i in range(n):
            s = sides[i]
            centers[i, 0] = np.random.uniform(s/2, 1-s/2)
            centers[i, 1] = np.random.uniform(s/2, 1-s/2)
        initial_configs.append((centers, sides))
    
    # Try optimization from each initial configuration
    for centers, sides in initial_configs:
        if time.time() > deadline - 30:
            break
        
        result = _try_optimization(centers, sides, max_iters=5000)
        if result is not None:
            c_opt, s_opt, sum_opt = result
            if sum_opt > best_sum:
                best_sum = sum_opt
                best_centers = c_opt
                best_sides = s_opt
    
    # If optimization didn't find anything, use the best initial config
    if best_centers is None:
        # Use the grid packing as fallback
        best_centers, best_sides = _grid_packing()
        best_sum = np.sum(best_sides)
    
    # Final validation
    if not _validity(best_centers, best_sides):
        # Fallback to grid packing
        best_centers, best_sides = _grid_packing()
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
