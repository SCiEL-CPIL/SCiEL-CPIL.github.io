import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _is_valid(circles):
    """Check if circles are disjoint and inside unit square."""
    n = len(circles)
    if n == 0:
        return True
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            if dist_sq < (min_dist - 1e-7) ** 2:
                return False
    return True

def _score(circles):
    """Compute sum of radii if valid, else 0."""
    if _is_valid(circles):
        return np.sum(circles[:, 2])
    return 0.0

def _objective(params, n_circles):
    """
    Objective: -sum(radii) + penalty for overlaps and boundary violations.
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    circles = params.reshape(n_circles, 3)
    penalty = 0.0
    
    # Boundary penalties
    # We want x - r >= 0 => x >= r => x - r >= 0
    # Penalty if x - r < 0 is (x - r)^2
    # Similarly for x + r <= 1 => 1 - (x + r) >= 0
    # Penalty if x + r > 1 is (x + r - 1)^2
    
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Vectorized boundary penalty
    # Left: x - r < 0
    pen_left = np.maximum(0, r - x) ** 2
    # Right: x + r > 1
    pen_right = np.maximum(0, x + r - 1) ** 2
    # Bottom: y - r < 0
    pen_bottom = np.maximum(0, r - y) ** 2
    # Top: y + r > 1
    pen_top = np.maximum(0, y + r - 1) ** 2
    # Radius < 0
    pen_r = np.maximum(0, -r) ** 2
    
    boundary_penalty = np.sum(pen_left + pen_right + pen_bottom + pen_top + pen_r)
    
    # Overlap penalties
    # dist^2 < (ri + rj)^2
    # We use squared overlap depth: (ri + rj - dist)^2 if positive
    # To be differentiable, we can just sum max(0, ri+rj-dist)^2
    # This is not differentiable at 0, but L-BFGS-B usually handles it okay, 
    # or we can use a smooth approximation. Let's try the direct one first.
    
    overlap_penalty = 0.0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            min_dist = ri + rj
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    # Weights
    # High penalty for violations to ensure validity, 
    # but we want to maximize radii, so the -sum(r) term should compete.
    # If penalty weight is too high, radii stay small.
    # If too low, circles overlap.
    # Let's use a large weight for penalty.
    W_BOUND = 1000.0
    W_OVERLAP = 1000.0
    
    return -np.sum(r) + W_BOUND * boundary_penalty + W_OVERLAP * overlap_penalty

def _make_initial_grid():
    """Create initial grid layout."""
    # 5x6 grid is 30, we need 26.
    # Let's use a 5x5 grid (25) + 1 in center? Or just 5x6 and remove 4?
    # Better: 5 rows, 6 cols, remove 4 corners?
    # Or just place 26 in a 5x6 grid, skipping 4 positions.
    
    rows, cols = 5, 6
    points = []
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            # Initial radius small
            r = 0.5 / max(rows, cols) * 0.5
            points.append([x, y, r])
            count += 1
        if count >= N:
            break
            
    return np.array(points)

def _make_initial_random(seed):
    """Create random initial layout."""
    np.random.seed(seed)
    circles = []
    for _ in range(N):
        r = np.random.uniform(0.02, 0.08)
        x = np.random.uniform(r, 1 - r)
        y = np.random.uniform(r, 1 - r)
        circles.append([x, y, r])
    return np.array(circles)

def _post_process(circles):
    """
    Ensure validity by slightly shrinking radii or moving centers if necessary.
    This is a safety net.
    """
    circles = circles.copy()
    # Check validity
    if _is_valid(circles):
        return circles
    
    # If invalid, shrink radii slightly
    # This is a crude fix, but ensures we return *something* valid if optimizer fails
    # Ideally, we want to avoid this.
    
    # Try to fix by reducing radii
    for _ in range(10):
        if _is_valid(circles):
            break
        circles[:, 2] *= 0.95 # Shrink radii by 5%
        
    # If still invalid, force valid positions
    if not _is_valid(circles):
        # Fallback to grid with very small radii
        grid = _make_initial_grid()
        grid[:, 2] = 0.01
        circles = grid
        
    return circles

def construct():
    """Main function to construct circle packing."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_circles = None
    best_score = 0.0
    
    # Initial configurations
    configs = [
        _make_initial_grid(),
        _make_initial_random(1),
        _make_initial_random(2),
        _make_initial_random(3),
        _make_initial_random(4)
    ]
    
    for config in configs:
        if time.time() > deadline:
            break
            
        # Ensure initial config is somewhat valid or at least not crazy
        # If initial is invalid, optimizer might struggle, but penalty method should handle it.
        
        params = config.flatten()
        
        try:
            result = minimize(
                _objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            optimized = result.x.reshape(N, 3)
            
            # Post-process to ensure validity
            optimized = _post_process(optimized)
            
            score = _score(optimized)
            if score > best_score:
                best_score = score
                best_circles = optimized.copy()
                
        except Exception as e:
            # If optimization fails, ignore this config
            continue
            
    if best_circles is None:
        # Fallback
        best_circles = _make_initial_grid()
        best_circles[:, 2] = 0.01
        
    return best_circles
