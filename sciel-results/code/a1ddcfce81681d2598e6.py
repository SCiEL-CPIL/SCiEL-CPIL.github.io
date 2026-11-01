"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _fast_covering_radius(points, probes):
    """
    Fast approximation of covering radius using probe points.
    points: (N, 3)
    probes: (M, 3)
    Returns max angular distance.
    """
    # Compute dot products: (M, N)
    dots = probes @ points.T
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Min distance corresponds to max dot product (closest points)
    # Wait, angular distance theta = arccos(dot).
    # Min theta -> Max dot.
    # We want the MAX theta, which is the MIN dot product? No.
    # For a specific probe y, the nearest point p minimizes theta(y, p).
    # So for each row y in probes, we find max dot product with any p.
    # Let d_max = max_p (y dot p).
    # Then theta_min = arccos(d_max).
    # We want max_y theta_min(y).
    # So we need min_y arccos(d_max(y)).
    # Since arccos is decreasing, this is equivalent to min_y d_max(y).
    # Let D = max_p (probes @ p). Shape (M,).
    # We want min(D).
    # Then radius = arccos(min(D)).
    
    max_dots = np.max(dots, axis=1)
    min_dot = np.min(max_dots)
    min_dot = np.clip(min_dot, -1.0, 1.0)
    return np.arccos(min_dot)

def _exact_covering_radius(points):
    """
    Compute exact covering radius using ConvexHull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # Invalid, return large value
    
    max_radius = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        denom = 2.0 * np.dot(cross, cross)
        
        if denom < 1e-12:
            continue
            
        a11 = np.dot(v1, v1)
        a12 = np.dot(v1, v2)
        a22 = np.dot(v2, v2)
        b1 = 0.5 * a11
        b2 = 0.5 * a22
        
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            continue
            
        t1 = (b1 * a22 - b2 * a12) / det
        t2 = (a11 * b2 - a12 * b1) / det
        
        O = p1 + t1 * v1 + t2 * v2
        norm_O = np.linalg.norm(O)
        if norm_O < 1e-10:
            continue
        C = O / norm_O
        
        cos_theta = np.clip(np.dot(C, p1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_radius:
            max_radius = theta
            
    return max_radius

def _sa_optimize(points, probes, time_limit=10.0):
    """
    Simulated Annealing optimization.
    """
    points = points.copy()
    current_radius = _fast_covering_radius(points, probes)
    best_points = points.copy()
    best_radius = current_radius
    
    T = 0.1 # Initial temperature (radians)
    cooling = 0.99
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < time_limit:
        # Move: perturb a random point
        idx = rng.randint(N)
        noise = rng.randn(3) * 0.1
        new_point = points[idx] + noise
        new_point /= np.linalg.norm(new_point)
        
        new_points = points.copy()
        new_points[idx] = new_point
        
        new_radius = _fast_covering_radius(new_points, probes)
        delta = new_radius - current_radius
        
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            points = new_points
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        T *= cooling
        if T < 1e-6:
            break
            
    return best_points, best_radius

def _local_search_facets(points, time_limit=20.0):
    """
    Local search based on convex hull facets.
    """
    points = points.copy()
    best_radius = _exact_covering_radius(points)
    start_time = time.time()
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        worst_simplex = None
        worst_radius = 0.0
        worst_center = None
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            denom = 2.0 * np.dot(cross, cross)
            if denom < 1e-12: continue
            
            a11 = np.dot(v1, v1)
            a12 = np.dot(v1, v2)
            a22 = np.dot(v2, v2)
            b1 = 0.5 * a11
            b2 = 0.5 * a22
            
            det = a11 * a22 - a12 * a12
            if abs(det) < 1e-12: continue
            
            t1 = (b1 * a22 - b2 * a12) / det
            t2 = (a11 * b2 - a12 * b1) / det
            
            O = p1 + t1 * v1 + t2 * v2
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-10: continue
            C = O / norm_O
            
            cos_theta = np.clip(np.dot(C, p1), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta > worst_radius:
                worst_radius = theta
                worst_simplex = simplex
                worst_center = C
                
        if worst_simplex is None:
            break
            
        new_points = points.copy()
        for idx in worst_simplex:
            p = points[idx]
            direction = worst_center - p
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction /= norm_dir
                p_new = p + step_size * direction
                new_points[idx] = p_new / np.linalg.norm(p_new)
                
        new_radius = _exact_covering_radius(new_points)
        
        if new_radius < best_radius:
            points = new_points
            best_radius = new_radius
            step_size *= 0.95
        else:
            step_size *= 0.8
            if step_size < 1e-5:
                break
                
    return points, best_radius

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    # Generate probe points for fast evaluation
    M = 5000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    probes = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random
    rng = np.random.RandomState(12345)
    for _ in range(5):
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
        
    best_points = None
    best_exact_radius = float('inf')
    
    for idx, init_pts in enumerate(configs):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5.0:
            break
            
        # Phase 1: SA to get a good candidate
        sa_time = min(15.0, remaining_time * 0.6)
        sa_pts, sa_radius = _sa_optimize(init_pts, probes, sa_time)
        
        # Phase 2: Local Search to refine
        ls_time = min(15.0, remaining_time - sa_time - 5.0)
        if ls_time > 1.0:
            final_pts, final_radius = _local_search_facets(sa_pts, ls_time)
        else:
            final_pts, final_radius = sa_pts, _exact_covering_radius(sa_pts)
            
        if final_radius < best_exact_radius:
            best_exact_radius = final_radius
            best_points = final_pts
            
    if best_points is None:
        best_points = configs[0]
        
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
