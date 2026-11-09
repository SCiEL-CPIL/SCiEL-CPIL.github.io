import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _compute_min_triangle_and_indices(points):
    """
    Compute the minimum triangle area and the indices of the triangle that forms it.
    """
    n = len(points)
    min_area = float('inf')
    min_idx = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0

    min_area, _ = _compute_min_triangle_and_indices(points)

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

def _halton_sequence(n, base):
    """
    Generate the first n terms of the Halton sequence in the given base.
    """
    result = []
    f = 1
    for i in range(1, n + 1):
        r = 0
        f = 1
        index = i
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        result.append(r)
    return np.array(result)

def _generate_initial_points():
    """
    Generate diverse initial configurations based on known optimal structures
    and low-discrepancy sequences.
    """
    configs = []
    
    # 1. Regular polygon (13 points)
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    configs.append(np.column_stack((np.cos(theta), np.sin(theta))))
    
    # 2. Halton sequence scaled to unit circle
    x = _halton_sequence(13, 2)
    y = _halton_sequence(13, 3)
    points = np.column_stack((x, y))
    # Center and scale
    points = (points - 0.5) * 2.0
    configs.append(points)
    
    # 3. Grid-like structure (approximating 7 outer, 6 inner)
    # Outer: 7 points on circle
    theta_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack((np.cos(theta_outer), np.sin(theta_outer)))
    # Inner: 6 points in a smaller hexagon
    theta_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack((0.5 * np.cos(theta_inner), 0.5 * np.sin(theta_inner)))
    configs.append(np.vstack([outer, inner]))
    
    # 4. Random with seed
    rng = np.random.default_rng(42)
    configs.append(rng.uniform(-1, 1, (13, 2)))
    
    # 5. Random on circle with jitter
    theta_rand = rng.uniform(0, 2 * np.pi, 13)
    r_rand = rng.uniform(0.8, 1.0, 13)
    configs.append(np.column_stack((r_rand * np.cos(theta_rand), r_rand * np.sin(theta_rand))))
    
    return configs

def _refine_with_subgradient(points, time_limit=10.0):
    """
    Refine points using subgradient ascent on the minimum triangle area.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        min_area, min_idx = _compute_min_triangle_and_indices(current_points)
        
        if min_area < 1e-10:
            # Collinear or nearly so, add random jitter
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
        
        i, j, k = min_idx
        p1, p2, p3 = current_points[i], current_points[j], current_points[k]
        
        # Subgradient of area with respect to p1, p2, p3
        # Area = 0.5 * | (p2 - p1) x (p3 - p1) |
        # Let v1 = p2 - p1, v2 = p3 - p1
        # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
        # Sign of cross product determines orientation
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        sign = np.sign(cross)
        
        # Gradient of area w.r.t p1: -0.5 * sign * (p3 - p2) rotated 90 deg
        # Gradient of area w.r.t p2:  0.5 * sign * (p3 - p1) rotated 90 deg
        # Gradient of area w.r.t p3:  0.5 * sign * (p1 - p2) rotated 90 deg
        
        # Rotate 90 deg: (x, y) -> (-y, x)
        def rot90(v):
            return np.array([-v[1], v[0]])
        
        grad_p1 = -0.5 * sign * rot90(p3 - p2)
        grad_p2 =  0.5 * sign * rot90(p3 - p1)
        grad_p3 =  0.5 * sign * rot90(p1 - p2)
        
        # Move points in direction of gradient (increase area)
        new_points = current_points.copy()
        new_points[i] = p1 + step_size * grad_p1
        new_points[j] = p2 + step_size * grad_p2
        new_points[k] = p3 + step_size * grad_p3
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-6:
                break
            if no_improve_count > max_no_improve:
                break
                
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Uses a hybrid approach:
    1. Generate diverse initial configurations.
    2. Apply subgradient refinement to each.
    3. Keep the best result.
    4. Use remaining time for stochastic restarts followed by refinement.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 min budget, 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Initial configurations
    configs = _generate_initial_points()
    
    # 2. Refine each initial config
    for cfg in configs:
        if time.time() > deadline:
            break
        refined_points, refined_score = _refine_with_subgradient(cfg, time_limit=5.0)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points

    # 3. Stochastic search with refinement
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        
        if restart_count % 5 == 0:
            # Random restart
            start_points = np.random.uniform(-1, 1, (13, 2))
        else:
            # Perturb best
            noise = np.random.normal(0, 0.5, (13, 2))
            start_points = best_points + noise
            
        refined_points, refined_score = _refine_with_subgradient(start_points, time_limit=3.0)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
            
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points
