import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 0.0

    # Calculate min triangle area
    # Vectorized approach for speed
    # i, j, k indices
    min_area = float('inf')
    # There are C(13,3) = 286 triangles
    # Loop is fast enough, but let's try to be efficient
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

def _get_min_triangle_info(points):
    """
    Returns (min_area, indices_tuple) for the smallest triangle.
    If there are ties, it returns the first one found.
    """
    min_area = float('inf')
    min_idx = None
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _expand_min_triangle(points, idx, step_size=0.05):
    """
    Attempts to increase the area of the triangle defined by idx.
    Moves the vertices away from the centroid of the triangle.
    Returns the modified points.
    """
    i, j, k = idx
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    centroid = (p1 + p2 + p3) / 3.0
    
    new_points = points.copy()
    
    # Move each vertex away from the centroid
    # This increases the size of the triangle
    # We use a small step to avoid overshooting or breaking other constraints
    new_points[i] = p1 + (p1 - centroid) * step_size
    new_points[j] = p2 + (p2 - centroid) * step_size
    new_points[k] = p3 + (p3 - centroid) * step_size
    
    return new_points

def _refine_local(points, max_iters=1000, time_limit=10.0):
    """
    Deterministic local refinement.
    1. Find min triangle.
    2. Expand it.
    3. Check if score improved.
    4. If not, try smaller steps or different moves (e.g., move one vertex perpendicular to opposite side).
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        min_area, min_idx = _get_min_triangle_info(current_points)
        
        if min_area < 1e-10:
            # Collinear or nearly so, need to break it
            # Random jitter might be better here
            current_points[min_idx[0]] += np.random.normal(0, 0.01, 2)
            new_score = _score(current_points)
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            continue

        # Try expanding the min triangle
        new_points = _expand_min_triangle(current_points, min_idx, step_size)
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            no_improve_count = 0
        else:
            # If expansion failed, try a different strategy:
            # Move one vertex perpendicular to the opposite side
            # This is a "shear" or "height increase" move
            i, j, k = min_idx
            p1, p2, p3 = current_points[i], current_points[j], current_points[k]
            
            # Vector from p2 to p3
            v23 = p3 - p2
            # Normal to v23
            n23 = np.array([-v23[1], v23[0]])
            n23_norm = np.linalg.norm(n23)
            if n23_norm > 1e-9:
                n23 = n23 / n23_norm
                
            # Determine direction to increase area (away from p1)
            # Cross product sign check
            # Area is proportional to | (p2-p1) x (p3-p1) |
            # Moving p1 along n23 (perp to p2-p3) changes height.
            
            # Try moving p1 in both normal directions
            for sign in [1, -1]:
                trial_points = current_points.copy()
                trial_points[i] = p1 + sign * n23 * step_size
                trial_score = _score(trial_points)
                
                if trial_score > current_score:
                    current_points = trial_points
                    current_score = trial_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
                    no_improve_count = 0
                    break
            else:
                # No improvement from simple moves
                no_improve_count += 1
                # Reduce step size
                step_size *= 0.9
                if step_size < 1e-5:
                    break
                if no_improve_count > max_no_improve:
                    break
                    
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Uses a hybrid approach:
    1. Generate diverse initial configurations (Halton, Regular Polygon, Random).
    2. Apply deterministic local refinement to each.
    3. Keep the best result.
    4. Use remaining time for stochastic "jumps" followed by refinement.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 min budget, 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Initial configurations
    configs = []
    
    # Regular polygon
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    configs.append(np.column_stack((np.cos(theta), np.sin(theta))))
    
    # Halton sequence (low discrepancy)
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_points = np.zeros((13, 2))
    for i in range(13):
        halton_points[i, 0] = halton(i + 1, 2)
        halton_points[i, 1] = halton(i + 1, 3)
    # Scale to unit circle-ish
    halton_points = (halton_points - 0.5) * 2.0
    configs.append(halton_points)
    
    # Random uniform in square
    rng = np.random.default_rng(42)
    configs.append(rng.uniform(-1, 1, (13, 2)))
    
    # Random points on circle with jitter
    theta_rand = rng.uniform(0, 2 * np.pi, 13)
    r_rand = rng.uniform(0.8, 1.0, 13)
    configs.append(np.column_stack((r_rand * np.cos(theta_rand), r_rand * np.sin(theta_rand))))
    
    # 2. Refine each initial config
    for cfg in configs:
        if time.time() > deadline:
            break
        refined_points, refined_score = _refine_local(cfg, max_iters=500, time_limit=5.0)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points

    # 3. Stochastic search with refinement
    # Use remaining time to perform random restarts/jumps
    # Each iteration:
    # a. Take best_points, add large noise
    # b. Refine
    # c. Update best
    
    # Estimate how many refinements we can do
    # _refine_local takes ~5s.
    # We have ~580s left.
    # So we can do ~100-150 restarts.
    
    restart_count = 0
    while time.time() < deadline - 10: # 10s buffer
        restart_count += 1
        # Generate a new candidate by perturbing the best known solution
        # Or a completely random one?
        # Perturbing best is usually better for local optima.
        # Let's mix: 80% perturb best, 20% random
        
        if restart_count % 5 == 0:
            # Random restart
            start_points = np.random.uniform(-1, 1, (13, 2))
        else:
            # Perturb best
            # Scale noise with time remaining? Or fixed?
            # Fixed large noise to escape local minima
            noise = np.random.normal(0, 0.5, (13, 2))
            start_points = best_points + noise
            
        # Refine
        refined_points, refined_score = _refine_local(start_points, max_iters=1000, time_limit=3.0)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
            # If we found a significantly better score, we might want to
            # increase the refinement depth for the next few iterations?
            # For now, keep it simple.
            
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        # Fallback: Regular polygon
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points
