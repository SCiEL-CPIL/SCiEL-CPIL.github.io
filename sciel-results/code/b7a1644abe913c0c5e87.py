import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _compute_min_triangle_and_indices(points):
    """
    Compute the minimum triangle area and the indices of the triangle achieving it.
    Returns (min_area, i, j, k)
    """
    n = len(points)
    min_area = float('inf')
    min_i, min_j, min_k = 0, 1, 2
    
    # Pre-computed indices for 13 points to avoid list generation overhead in loop
    # C(13,3) = 286
    # We'll just loop, it's fast enough for 13 points.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(pi[0] * (pj[1] - pk[1]) + pj[0] * (pk[1] - pi[1]) + pk[0] * (pi[1] - pj[1]))
                if area < min_area:
                    min_area = area
                    min_i, min_j, min_k = i, j, k
                    
    return min_area, min_i, min_j, min_k

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    min_area, _, _, _ = _compute_min_triangle_and_indices(points)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 1e-10:
        return 0.0
        
    return min_area / hull_area

def _halton_sequence(n, base):
    """Generate Halton sequence in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        while i > 0:
            f /= base
            i, digit = divmod(i, base)
            x += digit * f
        result.append(x)
    return result

def _generate_initial_points(seed_idx):
    """
    Generate diverse initial configurations.
    """
    rng = np.random.default_rng(seed_idx)
    
    # Strategy 1: Halton sequence mapped to a circle/square
    if seed_idx % 3 == 0:
        # Halton in 2D
        h1 = _halton_sequence(13, 2)
        h2 = _halton_sequence(13, 3)
        # Map to a circle for convexity
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        # Perturb angles slightly with Halton-based noise
        noise = np.array(h1) * 0.5 - 0.25
        angles += noise
        radii = 0.8 + 0.4 * np.array(h2)
        points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
    # Strategy 2: Regular polygon with radial jitter
    elif seed_idx % 3 == 1:
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        radii = 1.0 + rng.normal(0, 0.1, 13)
        points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
    # Strategy 3: Random points in a disk, then project to hull? 
    # Just random points in a square, then we'll let the optimizer fix it.
    else:
        points = rng.uniform(-1, 1, (13, 2))
        
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic geometric optimization with multistart.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    num_starts = 50
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
            
        points = _generate_initial_points(start_idx)
        
        # Initial score
        current_score = _score(points)
        
        # Local Refinement Loop
        # We iterate by finding the min triangle and pushing one of its vertices away.
        # Which vertex? The one that is "most interior" to the triangle or simply cycle through.
        # Better: Move the vertex that allows the largest increase in area for that specific triangle.
        
        max_iters = 2000
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            min_area, i, j, k = _compute_min_triangle_and_indices(points)
            
            # Check for collinearity issues
            if min_area < 1e-9:
                # Perturb randomly to escape
                idx = np.random.randint(13)
                points[idx] += np.random.normal(0, 0.05, 2)
                continue
            
            # We have triangle (i, j, k). We want to increase its area.
            # We can move i, j, or k.
            # Let's try moving each and pick the best improvement.
            # Actually, for speed, let's just move the one that is "most central" or random.
            # Let's try moving 'i' first, then 'j', then 'k' in subsequent iterations?
            # Or simply: Move the point that is closest to the line formed by the other two.
            
            # Distance from point p to line ab:
            def dist_to_line(p, a, b):
                # Area of triangle * 2 / length of base
                area = 0.5 * abs(a[0]*(b[1]-p[1]) + b[0]*(p[1]-a[1]) + p[0]*(a[1]-b[1]))
                length = np.linalg.norm(b - a)
                if length < 1e-9:
                    return 0
                return 2 * area / length
            
            d_i = dist_to_line(points[i], points[j], points[k])
            d_j = dist_to_line(points[j], points[i], points[k])
            d_k = dist_to_line(points[k], points[i], points[j])
            
            # Move the one with smallest distance (most "flat")
            if d_i <= d_j and d_i <= d_k:
                move_idx = i
                # Move away from the line jk
                # Normal vector to line jk
                v = points[k] - points[j]
                # Normal n = (-v_y, v_x)
                n = np.array([-v[1], v[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    n = n / n_norm
                    # Direction: away from the line. 
                    # Which side is 'i' on?
                    # Cross product of (v) and (i-j)
                    cross = v[0]*(points[i][1]-points[j][1]) - v[1]*(points[i][0]-points[j][0])
                    if cross > 0:
                        direction = n
                    else:
                        direction = -n
                    # Step size proportional to distance or fixed?
                    # Fixed small step
                    step = 0.05
                    points[move_idx] = points[move_idx] + direction * step
            elif d_j <= d_i and d_j <= d_k:
                move_idx = j
                v = points[k] - points[i]
                n = np.array([-v[1], v[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    n = n / n_norm
                    cross = v[0]*(points[j][1]-points[i][1]) - v[1]*(points[j][0]-points[i][0])
                    if cross > 0:
                        direction = n
                    else:
                        direction = -n
                    step = 0.05
                    points[move_idx] = points[move_idx] + direction * step
            else:
                move_idx = k
                v = points[j] - points[i]
                n = np.array([-v[1], v[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    n = n / n_norm
                    cross = v[0]*(points[k][1]-points[i][1]) - v[1]*(points[k][0]-points[i][0])
                    if cross > 0:
                        direction = n
                    else:
                        direction = -n
                    step = 0.05
                    points[move_idx] = points[move_idx] + direction * step
            
            # Optional: Normalize scale to prevent drift? 
            # The score is scale invariant. But if points drift far, numerical issues?
            # Let's center them occasionally.
            if it % 50 == 0:
                points -= np.mean(points, axis=0)
                
            # Calculate new score
            new_score = _score(points)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # If score didn't improve, maybe we got stuck in a local min for the min-triangle
                # Let's try a small random kick to escape
                if it % 10 == 0:
                    points[move_idx] += np.random.normal(0, 0.02, 2)
        
        # End of local refinement for this start
        
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        # Fallback: Regular polygon
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points
