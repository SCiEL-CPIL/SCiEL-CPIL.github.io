import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _compute_min_triangle_area(points):
    """
    Compute the minimum triangle area among all C(13,3) triangles.
    Returns min_area.
    """
    n = len(points)
    min_area = float('inf')
    
    # Unroll for speed? No, 286 is small. But let's use numpy vectorization for the area calc if possible?
    # Actually, pure python loop is fine for 13 points, but let's try to be efficient.
    # We can compute all areas in a vectorized way if we precompute cross products?
    # Let's stick to a tight loop for clarity and low overhead per call, 
    # but we will call this frequently.
    
    # Optimization: Pre-compute indices
    # C(13, 3) = 286
    # We will just loop.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                # This is 0.5 * |cross(pj-pi, pk-pi)|
                # cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    # Early exit if area is tiny? No, we need the true min for score.
                    
    return min_area

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    min_area = _compute_min_triangle_area(points)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 1e-10:
        return 0.0
        
    return min_area / hull_area

def _project_to_hull_boundary(points, idx, rng):
    """
    If a point is interior, push it out to the hull boundary or expand the hull.
    This helps maintain a "convex-like" spread which is often optimal for Heilbronn.
    """
    # Simple heuristic: if the point is not on the hull, move it radially outward from the centroid
    centroid = np.mean(points, axis=0)
    p = points[idx]
    vec = p - centroid
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        vec = rng.normal(0, 1, 2)
        norm = np.linalg.norm(vec)
    
    # Determine current max radius
    radii = np.linalg.norm(points - centroid, axis=1)
    max_r = np.max(radii)
    
    # Move to a radius slightly larger than max_r or randomize
    new_r = max_r * (1.0 + rng.uniform(0, 0.5))
    new_p = centroid + (vec / norm) * new_r
    
    return new_p

def heilbronn_convex13() -> np.ndarray:
    """
    Simulated Annealing with structural shocks for Heilbronn problem.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial configuration: Regular polygon is a strong start
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    current_points = np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Normalize to unit area? Score is scale invariant, so we can work in any scale.
    # Let's keep it in [0,1] or centered.
    current_points -= np.mean(current_points, axis=0)
    
    current_score = _score(current_points)
    if current_score > best_score:
        best_score = current_score
        best_points = current_points.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.9995
    max_iters = 50000
    stall_counter = 0
    stall_limit = 500
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # 1. Generate a perturbation
        # Strategy: 
        # - 80% of the time: Small random move of a random point (local refinement)
        # - 20% of the time: "Shock" - move a random point significantly (escape local min)
        
        move_idx = rng.integers(0, 13)
        
        if rng.random() < 0.8:
            # Local move: Gaussian noise
            step_size = 0.1 * T  # Scale step with temperature? Or fixed?
            # Fixed small step is better for fine tuning, but T helps exploration.
            # Let's use a mix.
            noise = rng.normal(0, 0.05, 2)
            new_points = current_points.copy()
            new_points[move_idx] += noise
        else:
            # Shock move: 
            # Option A: Push to hull boundary
            # Option B: Randomize one coordinate significantly
            if rng.random() < 0.5:
                new_points = current_points.copy()
                new_points[move_idx] = _project_to_hull_boundary(current_points, move_idx, rng)
            else:
                new_points = current_points.copy()
                # Large random jump
                new_points[move_idx] = current_points[move_idx] + rng.normal(0, 0.5, 2)
        
        # 2. Evaluate new state
        new_score = _score(new_points)
        
        # 3. Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            current_points = new_points
            stall_counter = 0
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Metropolis criterion: accept with probability exp((new - current) / T)
            # Note: We want to maximize score.
            # delta = new_score - current_score (negative)
            delta = new_score - current_score
            if delta < 0:
                # Accept if random < exp(delta / T)
                # Since delta is negative, exp(delta/T) < 1
                # We must be careful with T=0
                if T > 1e-9:
                    accept_prob = np.exp(delta / T)
                    if rng.random() < accept_prob:
                        current_points = new_points
                        current_score = new_score
                        stall_counter += 1
                    else:
                        stall_counter += 1
                else:
                    stall_counter += 1
            
            # If we are stuck for a long time, force a larger exploration
            if stall_counter > stall_limit:
                # Reset temperature or force a big jump
                # Let's do a big jump on a random point
                big_idx = rng.integers(0, 13)
                current_points[big_idx] = _project_to_hull_boundary(current_points, big_idx, rng)
                current_score = _score(current_points)
                stall_counter = 0
                # Don't update best unless it's better
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Reheat?
                # If we've cooled down and are still stuck, reheat
                if stall_counter > stall_limit * 2:
                    T = 0.5
                    stall_counter = 0
        
        # Periodic re-centering to prevent drift
        if it % 100 == 0:
            current_points -= np.mean(current_points, axis=0)
            # Note: Re-centering doesn't change the score (scale/translation invariant), 
            # but it keeps numbers small. We don't need to re-evaluate score here strictly,
            # but if we changed points, we should update current_points.
            # Actually, re-centering is just a translation. Score is invariant.
            # So we can just do it.
            
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        # Fallback: Regular polygon
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points
