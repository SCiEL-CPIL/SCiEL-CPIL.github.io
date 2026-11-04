import time
import numpy as np
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0.0 if the point set is degenerate or invalid.
    """
    points = np.asarray(points)
    if points.shape[0] < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # Calculate minimum triangle area
    min_area = float('inf')
    n = len(points)
    
    # Vectorized calculation for all triples
    # There are C(n, 3) triangles
    # For n=13, C(13,3) = 286, which is small enough to loop or vectorize simply
    
    # Using itertools for clarity and speed for small n
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    if min_area <= 1e-12:
        return 0.0
        
    return min_area / hull_area

def _random_points(rng):
    """Generate random points in the unit square."""
    return np.array([[rng.random(), rng.random()] for _ in range(N)])

def _hill_climb(points, rng, max_iter=10000, step_size=0.1, time_limit=None):
    """
    Local optimization using a simple random perturbation hill climbing.
    """
    best_points = points.copy()
    best_score = _compute_score(best_points)
    current_points = points.copy()
    current_score = best_score
    
    start_time = time.time()
    
    for i in range(max_iter):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        # Perturb a random point
        new_points = current_points.copy()
        idx = rng.randint(0, N - 1)
        # Gaussian noise
        noise = rng.gauss(0, step_size)
        new_points[idx, 0] += noise
        new_points[idx, 1] += rng.gauss(0, step_size)
        
        # Optional: clamp to [0, 1] or allow out of bounds? 
        # Convex hull area changes if points move, so we don't strictly need to clamp 
        # unless we want to keep them in a specific domain. 
        # The problem says "normalized to unit area", but the interface says 
        # "score is the minimum triangle area divided by the convex hull area".
        # This ratio is scale-invariant. So we can let points roam freely.
        # However, staying in a reasonable range helps numerical stability.
        # Let's clamp to [-10, 10] to avoid overflow.
        new_points = np.clip(new_points, -10, 10)
        
        new_score = _compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = new_points.copy()
                best_score = current_score
            # Increase step size slightly on success? Or keep constant.
            # Let's keep it simple.
        else:
            # If it fails, we could decrease step size or just reject.
            # Simple rejection.
            pass
            
    return best_points, best_score

def _simulated_annealing(points, rng, start_temp=1.0, end_temp=0.01, max_iter=5000, step_size=0.5, time_limit=None):
    """
    Simulated annealing for global search.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    for i in range(max_iter):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (i / max_iter)
        
        # Perturb
        new_points = current_points.copy()
        idx = rng.randint(0, N - 1)
        new_points[idx, 0] += rng.gauss(0, step_size)
        new_points[idx, 1] += rng.gauss(0, step_size)
        new_points = np.clip(new_points, -10, 10)
        
        new_score = _compute_score(new_points)
        
        delta_score = new_score - current_score
        
        if delta_score > 0:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = new_points.copy()
                best_score = current_score
        else:
            # Accept with probability exp(delta_score / temp)
            # Since scores are small, delta_score is small. 
            # We might need to scale the temperature.
            # Let's assume scores are in range [0, 1].
            # If temp is 1.0, exp(-0.01) ~ 0.99.
            # If temp is 0.01, exp(-1.0) ~ 0.37.
            if rng.random() < np.exp(delta_score / max(temp, 1e-9)):
                current_points = new_points
                current_score = new_score
                
    return best_points, best_score

def construct():
    rng = np.random.RandomState(42)
    
    # Deadline guard: 150s limit, leave 10s margin -> 140s budget
    deadline = time.time() + 140
    
    best_points = None
    best_score = 0.0
    
    # 1. Try a few random restarts with simulated annealing
    # and hill climbing.
    
    # Initial guess: regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    init_points = np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))
    
    # Scale to unit area? 
    # Regular polygon area is approx pi * r^2. 
    # But the score is ratio, so scale doesn't matter.
    # However, starting with a decent shape helps.
    
    # Let's try optimizing the regular polygon vertices slightly
    # Actually, let's just start with random points and SA.
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate random starting points
        # Mix of uniform in square and random polygon
        if r % 2 == 0:
            start_pts = _random_points(rng)
        else:
            # Random convex polygon
            angles = np.sort(rng.random(N) * 2 * np.pi)
            radii = rng.uniform(0.5, 1.0, N)
            start_pts = np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))
            
        # Simulated Annealing
        sa_pts, sa_score = _simulated_annealing(
            start_pts, rng, 
            start_temp=0.5, end_temp=0.001, 
            max_iter=2000, step_size=0.5,
            time_limit=min(10, max(0, deadline - time.time() - 5))
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
            
        # Hill Climbing from SA result
        hc_pts, hc_score = _hill_climb(
            sa_pts, rng,
            max_iter=2000, step_size=0.1,
            time_limit=min(10, max(0, deadline - time.time() - 5))
        )
        
        if hc_score > best_score:
            best_score = hc_score
            best_points = hc_pts.copy()
            
    # Final polish: very small steps hill climb
    if best_points is not None and time.time() < deadline - 5:
        final_pts, final_score = _hill_climb(
            best_points, rng,
            max_iter=5000, step_size=0.01,
            time_limit=min(10, max(0, deadline - time.time() - 2))
        )
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
            
    if best_points is None:
        # Fallback
        best_points = _random_points(rng)
        
    return best_points

# EVOLVE-BLOCK-END
