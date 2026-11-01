"""Circle packing in perimeter-4 rectangle: maximize sum of 21 radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21
DEADLINE = 140.0  # wall-clock guard with safety margin before 150s kill

def _validate_and_score(circles):
    """
    circles: (n, 3) array of [x, y, r]
    Returns (value, valid) where value = sum of radii if valid, else -inf.
    """
    if len(circles) != N:
        return -np.inf, False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # All radii must be positive
    if np.any(rs <= 0):
        return -np.inf, False
    
    # Compute bounding box
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    
    w = xmax - xmin
    h = ymax - ymin
    
    if w < 0 or h < 0:
        return -np.inf, False
    
    perimeter = 2 * (w + h)
    if perimeter > 4.0 + 1e-9:
        return -np.inf, False
    
    # Check disjoint interiors: distance between centers >= r_i + r_j
    # Use pairwise distances
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return -np.inf, False
    
    return np.sum(rs), True

def _score_with_penalty(circles, penalty=1e6):
    """
    Score = sum(r) - penalty * max(0, perimeter - 4) - penalty * max_overlap_violation
    This is a smooth-ish objective for optimization.
    """
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    n = len(circles)
    
    # Bounding box
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    
    w = xmax - xmin
    h = ymax - ymin
    
    perimeter = 2 * (w + h)
    perim_violation = max(0.0, perimeter - 4.0)
    
    # Overlap violations
    max_overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > max_overlap:
                max_overlap = overlap
    
    return np.sum(rs) - penalty * perim_violation - penalty * max_overlap

def _construct_grid_initial():
    """Create a grid-based initial configuration."""
    # Try different grid layouts
    best = None
    best_score = -np.inf
    
    for cols in range(3, 8):
        rows = (N + cols - 1) // cols
        # Rectangle with perimeter 4: 2*(w+h) = 4 => w+h = 2
        # For a grid of cols x rows, we want roughly equal spacing
        # Let's set w = 2 * cols/(cols+rows), h = 2 * rows/(cols+rows)
        w = 2.0 * cols / (cols + rows)
        h = 2.0 * rows / (cols + rows)
        
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) >= N:
                    break
                x = (i + 0.5) * w / cols
                y = (j + 0.5) * h / rows
                r = min(w / cols, h / rows) * 0.45
                circles.append([x, y, r])
        
        if len(circles) < N:
            continue
        
        circles = np.array(circles[:N])
        val, valid = _validate_and_score(circles)
        if valid and val > best_score:
            best_score = val
            best = circles
    
    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
    
    return best

def _local_search(circles, max_iters=200, step=0.01, directions=8):
    """
    Local search: try moving circles in 8 directions with adaptive step size.
    """
    current = circles.copy()
    current_score, current_valid = _validate_and_score(current)
    if not current_valid:
        current_score = -np.inf
    
    best = current.copy()
    best_score = current_score
    
    angles = np.linspace(0, 2 * np.pi, directions, endpoint=False)
    step_sizes = [step, step * 0.5, step * 0.1, step * 2.0]
    
    for it in range(max_iters):
        improved = False
        for idx in range(N):
            for angle in angles:
                for ss in step_sizes:
                    candidate = current.copy()
                    candidate[idx, 0] += ss * np.cos(angle)
                    candidate[idx, 1] += ss * np.sin(angle)
                    
                    val, valid = _validate_and_score(candidate)
                    if valid and val > best_score + 1e-12:
                        best = candidate.copy()
                        best_score = val
                        current = candidate.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def _expand_radii(circles, max_iters=100):
    """
    Try to expand radii while maintaining disjointness and perimeter constraint.
    """
    current = circles.copy()
    current_score, current_valid = _validate_and_score(current)
    if not current_valid:
        return current, current_score
    
    best = current.copy()
    best_score = current_score
    
    for it in range(max_iters):
        improved = False
        for idx in range(N):
            # Try increasing radius slightly
            for factor in [1.01, 1.02, 1.05]:
                candidate = current.copy()
                candidate[idx, 2] *= factor
                
                val, valid = _validate_and_score(candidate)
                if valid and val > best_score + 1e-12:
                    best = candidate.copy()
                    best_score = val
                    current = candidate.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best, best_score

def _relax_positions(circles, max_iters=50):
    """
    Relax positions to reduce overlaps while keeping radii fixed.
    """
    current = circles.copy()
    
    for it in range(max_iters):
        # Check for overlaps
        overlaps = []
        xs = current[:, 0]
        ys = current[:, 1]
        rs = current[:, 2]
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = rs[i] + rs[j] - dist
                if overlap > 1e-12:
                    overlaps.append((i, j, overlap, dx, dy, dist))
        
        if not overlaps:
            break
        
        # Move circles apart
        for i, j, overlap, dx, dy, dist in overlaps:
            if dist < 1e-12:
                # Random nudge
                angle = np.random.rand() * 2 * np.pi
                current[i, 0] += overlap * 0.5 * np.cos(angle)
                current[i, 1] += overlap * 0.5 * np.sin(angle)
                current[j, 0] -= overlap * 0.5 * np.cos(angle)
                current[j, 1] -= overlap * 0.5 * np.sin(angle)
            else:
                # Move along the line connecting centers
                ux = dx / dist
                uy = dy / dist
                current[i, 0] += overlap * 0.5 * ux
                current[i, 1] += overlap * 0.5 * uy
                current[j, 0] -= overlap * 0.5 * ux
                current[j, 1] -= overlap * 0.5 * uy
    
    return current

def _smooth_objective(x, circles_init):
    """
    Smooth objective for scipy optimization.
    x: flattened array of [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    circles = x.reshape(N, 3)
    
    # Ensure radii are positive
    rs = np.maximum(circles[:, 2], 1e-6)
    circles[:, 2] = rs
    
    xs = circles[:, 0]
    ys = circles[:, 1]
    
    # Bounding box
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    
    w = xmax - xmin
    h = ymax - ymin
    
    perimeter = 2 * (w + h)
    
    # Smooth perimeter penalty using soft-max
    # We want perimeter <= 4
    perim_penalty = 0.0
    if perimeter > 4.0:
        perim_penalty = (perimeter - 4.0) ** 2 * 100
    
    # Smooth overlap penalty
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2 * 100
    
    # Objective: maximize sum of radii, minimize penalties
    return -(np.sum(rs)) + perim_penalty + overlap_penalty

def _scipy_optimize(circles, max_iters=1000):
    """
    Use scipy to optimize the smooth objective.
    """
    x0 = circles.flatten()
    
    result = minimize(
        _smooth_objective,
        x0,
        args=(circles,),
        method='L-BFGS-B',
        options={'maxiter': max_iters, 'ftol': 1e-10}
    )
    
    optimized = result.x.reshape(N, 3)
    # Ensure radii are positive
    optimized[:, 2] = np.maximum(optimized[:, 2], 1e-6)
    
    return optimized

def construct():
    start_time = time.time()
    
    # Initial configuration
    circles = _construct_grid_initial()
    best_circles = circles.copy()
    best_score, best_valid = _validate_and_score(best_circles)
    if not best_valid:
        best_score = -np.inf
    
    # Iterative improvement loop
    while time.time() - start_time < DEADLINE - 10:
        # 1. Local search on positions
        circles, score = _local_search(circles, max_iters=100, step=0.02)
        
        # 2. Expand radii
        circles, score = _expand_radii(circles, max_iters=50)
        
        # 3. Relax positions to fix any overlaps
        circles = _relax_positions(circles, max_iters=30)
        
        # 4. Scipy optimization
        circles = _scipy_optimize(circles, max_iters=200)
        
        # Validate
        val, valid = _validate_and_score(circles)
        if valid and val > best_score:
            best_score = val
            best_circles = circles.copy()
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Final validation
    final_val, final_valid = _validate_and_score(best_circles)
    if not final_valid:
        # Fallback to a simple valid configuration
        best_circles = _construct_grid_initial()
    
    return best_circles

# EVOLVE-BLOCK-START
# (The construct function is defined above, so we just need to ensure it's accessible)
# EVOLVE-BLOCK-END
