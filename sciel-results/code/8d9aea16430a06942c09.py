"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _spherical_circumcenter(a, b, c):
    """Compute the spherical circumcenter of triangle (a,b,c) on the unit sphere.
    Returns the point on the sphere equidistant from a, b, c (closest to the triangle)."""
    # The circumcenter lies on the line through the origin in the direction of
    # the cross product of (b-a) and (c-a), normalized.
    # More robustly: solve for x on unit sphere such that x.a = x.b = x.c
    # This means x is orthogonal to (b-a) and (c-a), so x is parallel to cross(b-a, c-a)
    u = b - a
    v = c - a
    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-12:
        # Degenerate triangle; return midpoint of a and b projected
        m = a + b
        m_norm = np.linalg.norm(m)
        if m_norm < 1e-12:
            return a.copy()
        return m / m_norm
    
    center = w / w_norm
    # Choose the sign so that the center is on the same side as the triangle
    # (i.e., dot product with a is positive for a small triangle)
    if np.dot(center, a) < 0:
        center = -center
    return center


def _covering_radius_and_worst(points):
    """Compute covering radius in degrees and identify the worst facet.
    Returns (radius_degrees, worst_facet_indices, worst_circumcenter)."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 3):
        return np.inf, None, None
    if not np.all(np.isfinite(points)):
        return np.inf, None, None
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return np.inf, None, None
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf, None, None
    
    max_rad = 0.0
    worst_indices = None
    worst_center = None
    
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        center = _spherical_circumcenter(a, b, c)
        
        # Angular distance from center to a
        cos_dist = np.clip(np.dot(center, a), -1.0, 1.0)
        ang_dist = np.arccos(cos_dist) * 180.0 / np.pi
        
        if ang_dist > max_rad:
            max_rad = ang_dist
            worst_indices = simplex.copy()
            worst_center = center
    
    return max_rad, worst_indices, worst_center


def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on the sphere using the Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate n random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _optimize_from_start(points, max_time=60.0, tol=1e-8):
    """Iteratively optimize a configuration to minimize covering radius.
    
    Strategy: At each iteration, find the worst facet, compute its circumcenter,
    and move the three vertices of that facet toward the circumcenter using
    a small step. Use line search to ensure improvement.
    """
    points = _project_to_sphere(points.copy())
    start_time = time.time()
    
    # First compute initial radius
    current_rad, worst_idx, worst_center = _covering_radius_and_worst(points)
    if not np.isfinite(current_rad):
        return points
    
    best_points = points.copy()
    best_rad = current_rad
    
    step_size = 0.1  # Initial step in radians
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(5000):
        if time.time() - start_time > max_time:
            break
        if current_rad < tol:
            break
        if no_improve_count > max_no_improve:
            break
        
        # Find the worst facet
        current_rad, worst_idx, worst_center = _covering_radius_and_worst(points)
        if worst_idx is None or not np.isfinite(current_rad):
            break
        
        # Try to improve by moving vertices of the worst facet
        improved = False
        for trial in range(3):
            if time.time() - start_time > max_time:
                break
            
            new_points = points.copy()
            for vi in worst_idx:
                # Move point vi toward worst_center
                direction = worst_center - points[vi]
                dir_norm = np.linalg.norm(direction)
                if dir_norm < 1e-12:
                    continue
                direction = direction / dir_norm
                new_points[vi] = points[vi] + step_size * direction
                new_points[vi] = new_points[vi] / np.linalg.norm(new_points[vi])
            
            new_rad, _, _ = _covering_radius_and_worst(new_points)
            
            if np.isfinite(new_rad) and new_rad < current_rad - 1e-12:
                points = new_points
                current_rad = new_rad
                improved = True
                no_improve_count = 0
                if current_rad < best_rad:
                    best_rad = current_rad
                    best_points = points.copy()
                break
            else:
                # Reduce step size
                step_size *= 0.5
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best_points


def _multi_start_optimize(deadline=120.0):
    """Try multiple starting configurations and return the best one."""
    start_time = time.time()
    
    # Generate multiple starting configurations
    starts = []
    
    # Fibonacci sphere with different offsets
    for offset in [0.3, 0.5, 0.7]:
        starts.append(_fibonacci_sphere(N, offset))
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789]:
        starts.append(_random_sphere(N, seed))
    
    # Spherical code inspired start: icosahedron-like symmetry
    # Create a base icosahedron and subdivide
    def _icosahedron_points():
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        pts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        ])
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms
    
    ico = _icosahedron_points()
    # Add more points by rotating and combining
    from scipy.spatial.transform import Rotation
    angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    rot_pts = []
    for ang in angles:
        rot = Rotation.from_euler('z', ang)
        rot_pts.append(rot.apply(ico))
    # Take first N points from the union
    all_pts = np.vstack(rot_pts)
    # Normalize
    all_pts = _project_to_sphere(all_pts)
    # Remove duplicates and take first N
    unique_pts = []
    for p in all_pts:
        if all(np.linalg.norm(p - q) > 1e-6 for q in unique_pts):
            unique_pts.append(p)
        if len(unique_pts) >= N:
            break
    if len(unique_pts) >= N:
        starts.append(np.array(unique_pts[:N]))
    
    # Optimize each start with a time budget
    best_points = None
    best_rad = np.inf
    
    for i, start_pts in enumerate(starts):
        if time.time() - start_time > deadline:
            break
        
        # Allocate time for this optimization
        remaining = deadline - (time.time() - start_time)
        time_budget = min(remaining * 0.8, 40.0)  # Use at most 40s per start
        if time_budget < 5.0:
            break
        
        optimized = _optimize_from_start(start_pts, max_time=time_budget)
        rad, _, _ = _covering_radius_and_worst(optimized)
        
        if np.isfinite(rad) and rad < best_rad:
            best_rad = rad
            best_points = optimized.copy()
    
    if best_points is None:
        # Fallback to Fibonacci
        best_points = _fibonacci_sphere(N)
    
    return best_points


def construct():
    start_time = time.time()
    deadline = 130.0  # Leave margin before the 150s kill
    
    points = _multi_start_optimize(deadline=deadline)
    
    # Final check
    rad, _, _ = _covering_radius_and_worst(points)
    if not np.isfinite(rad):
        # Fallback
        points = _fibonacci_sphere(N)
    
    return points
# EVOLVE-BLOCK-END
