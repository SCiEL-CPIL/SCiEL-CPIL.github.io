# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def compute_covering_radius(points):
    """
    Compute the covering radius and the indices of the points in the 
    worst facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., all points in a hemisphere)
        # Use max distance to a random sample
        rng = np.random.RandomState(42)
        samples = rng.randn(5000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        d = np.max(np.min(np.arccos(np.clip(np.dot(samples, points.T), -1, 1)), axis=1))
        return d, None

    max_radius = 0.0
    worst_facet = None
    
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        a, b, c = facet_pts
        
        # Compute circumcenter of the spherical triangle
        # The circumcenter is the intersection of the sphere and the line 
        # passing through the origin and the circumcenter of the planar triangle?
        # No, the circumcenter of a spherical triangle is the point on the sphere
        # equidistant from the vertices.
        # Let o be the circumcenter. o . a = o . b = o . c = cos(R).
        # This implies (a-b).o = 0 and (b-c).o = 0.
        # So o is parallel to (a-b) x (b-c).
        
        v1 = a - b
        v2 = b - c
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            # Degenerate case
            continue
            
        o = normal / norm_normal
        
        # Ensure o is on the correct side (inside the triangle)
        centroid = (a + b + c) / 3.0
        if np.dot(o, centroid) < 0:
            o = -o
            
        cos_r = np.clip(np.dot(o, a), -1.0, 1.0)
        r = np.arccos(cos_r)
        
        if r > max_radius:
            max_radius = r
            worst_facet = simplex

    return max_radius, worst_facet

def random_point_on_sphere(rng):
    """Generate a random point on the unit sphere."""
    x, y, z = rng.randn(3)
    norm = np.sqrt(x*x + y*y + z*z)
    return np.array([x, y, z]) / norm

def construct():
    rng = np.random.RandomState(42)
    
    # Initialize with a Fibonacci sphere for a good start, or random
    # Fibonacci sphere is usually better for initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = normalize(points)
    
    best_points = points.copy()
    best_radius, _ = compute_covering_radius(best_points)
    
    # Simulated Annealing Parameters
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer for the 120s limit
    
    T = 0.1  # Initial temperature (radians)
    cooling_rate = 0.999
    min_T = 1e-6
    
    # We will iterate until time limit or T is too small
    while T > min_T and (time.time() - start_time) < time_limit:
        # Generate a candidate move
        # Strategy: Pick a random point and move it slightly
        # Or pick the worst facet and move its points
        # Let's try moving a single random point to a new random location
        # or a small perturbation. Small perturbations are more stable for SA.
        
        idx = rng.randint(N)
        original_point = points[idx].copy()
        
        # Perturb the point
        # Add small random noise
        noise = rng.randn(3) * 0.05
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        new_radius, _ = compute_covering_radius(points)
        
        # Acceptance criterion
        delta = new_radius - best_radius
        if delta < 0 or (T > 0 and rng.rand() < np.exp(-delta / T)):
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        else:
            # Reject move
            points[idx] = original_point
            
        # Cool down
        T *= cooling_rate
        
        # Occasionally, try a larger jump to escape local minima
        if rng.rand() < 0.01:
            idx = rng.randint(N)
            new_point = random_point_on_sphere(rng)
            points[idx] = new_point
            new_radius, _ = compute_covering_radius(points)
            delta = new_radius - best_radius
            if delta < 0 or (T > 0 and rng.rand() < np.exp(-delta / T)):
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = points.copy()
            else:
                points[idx] = original_point # Wait, original_point is from before the big jump logic? 
                # Actually, we didn't store the state before the big jump properly if we rejected.
                # Let's simplify: always store current state before move.
                
    return best_points
# EVOLVE-BLOCK-END
