"""Improved program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START


def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    
    The covering radius equals the largest circumradius of the facets of the convex hull.
    """
    points = np.asarray(points, dtype=np.float64)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        # Fallback: use nearest-neighbor based estimate
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        n_samples = 200000
        u = np.random.randn(n_samples, 3)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        _, idx = tree.query(u)
        dists = np.arccos(np.clip(np.sum(u * points[idx], axis=1), -1, 1))
        return np.degrees(np.max(dists))
    
    max_radius = 0.0
    for simplex in simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        # Compute the plane containing p0, p1, p2
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate triangle - use max pairwise distance / 2 as estimate
            d01 = np.arccos(np.clip(np.dot(p0, p1), -1, 1))
            d12 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
            d02 = np.arccos(np.clip(np.dot(p0, p2), -1, 1))
            R = max(d01, d12, d02) / 2.0
            max_radius = max(max_radius, R)
            continue
        n = n / n_norm
        
        # Distance from origin to the plane
        d = np.dot(p0, n)
        d = np.clip(d, -1.0, 1.0)
        
        # Radius of the circle in 3D
        r_3d_sq = max(0.0, 1.0 - d * d)
        r_3d = np.sqrt(r_3d_sq)
        
        # Angular circumradius
        if r_3d < 1e-12:
            R = 0.0
        else:
            R = np.arcsin(np.clip(r_3d, -1.0, 1.0))
        
        max_radius = max(max_radius, R)
    
    return np.degrees(max_radius)


def _fast_covering_radius(points, tree_cache=None):
    """Faster covering radius estimate using sampling. Used during optimization."""
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    
    # Use a fixed set of test points for speed
    if tree_cache is None:
        n_samples = 50000
        u = np.random.randn(n_samples, 3)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        tree_cache = (tree, u)
    else:
        tree, u = tree_cache
    
    _, idx = tree.query(u)
    dists = np.arccos(np.clip(np.sum(u * points[idx], axis=1), -1, 1))
    return np.degrees(np.max(dists))


def _fibonacci_start():
    """Generate a Fibonacci spiral configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start():
    """Generate a random configuration of N points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add random perturbation
    noise = np.random.randn(N, 3) * 0.15
    points += noise
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _hill_climb(points, max_time=None, verbose=False):
    """Perform hill-climbing optimization to minimize the covering radius."""
    points = points.copy()
    n = len(points)
    
    start_time = time.time()
    
    # Precompute test points for fast evaluation
    n_samples = 50000
    test_points = np.random.randn(n_samples, 3)
    test_points /= np.linalg.norm(test_points, axis=1, keepdims=True)
    
    best_radius = _covering_radius(points)
    best_points = points.copy()
    
    # Initial step size
    step = 0.05
    
    while True:
        if max_time is not None and time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE - 5:
            break
        
        improved = False
        
        for _ in range(100):  # Inner iterations
            if time.time() > DEADLINE - 5:
                break
            
            # Pick a random point to move
            i = np.random.randint(n)
            orig = points[i].copy()
            
            # Try moving in a random direction
            for _ in range(15):
                # Random direction on the sphere
                d = np.random.randn(3)
                d /= np.linalg.norm(d)
                
                # Move point i in direction d by step
                new_point = points[i] + step * d
                new_point /= np.linalg.norm(new_point)
                points[i] = new_point
                
                # Quick check using fast radius
                r_fast = _fast_covering_radius(points, (None, test_points))
                
                if r_fast < best_radius - 0.005:
                    # Full check
                    r_full = _covering_radius(points)
                    if r_full < best_radius - 1e-9:
                        best_radius = r_full
                        best_points = points.copy()
                        improved = True
                        break
                    else:
                        points[i] = orig
                        break
                else:
                    points[i] = orig
            
            if improved:
                # Reduce step size slightly
                step *= 0.95
                break
        
        if not improved:
            # Annealing: occasionally increase step size to escape local minima
            if step < 0.02:
                step = min(step * 1.5, 0.1)
            elif np.random.rand() < 0.3:
                step = max(step * 0.8, 0.005)
            else:
                step *= 0.9
            
            if step < 1e-6:
                break
    
    return best_points, best_radius


def _refine_with_gradient(points, max_time=None):
    """Refine points using a gradient-like approach focusing on worst facets."""
    points = points.copy()
    n = len(points)
    
    start_time = time.time()
    best_radius = _covering_radius(points)
    best_points = points.copy()
    
    step = 0.01
    
    while True:
        if max_time is not None and time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE - 5:
            break
        
        # Compute covering radius and identify worst facet
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
        except Exception:
            break
        
        max_r = 0.0
        worst_simplex = None
        for simplex in simplices:
            p0 = points[simplex[0]]
            p1 = points[simplex[1]]
            p2 = points[simplex[2]]
            
            n_vec = np.cross(p1 - p0, p2 - p0)
            n_norm = np.linalg.norm(n_vec)
            if n_norm < 1e-12:
                continue
            n_vec = n_vec / n_norm
            d = np.clip(np.dot(p0, n_vec), -1.0, 1.0)
            r_3d = np.sqrt(max(0.0, 1.0 - d * d))
            R = np.arcsin(np.clip(r_3d, -1.0, 1.0))
            
            if R > max_r:
                max_r = R
                worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Move points in the worst facet to reduce the circumradius
        # The circumcenter is in the direction of the normal
        p0 = points[worst_simplex[0]]
        p1 = points[worst_simplex[1]]
        p2 = points[worst_simplex[2]]
        
        n_vec = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n_vec)
        if n_norm < 1e-12:
            break
        n_vec = n_vec / n_norm
        
        # Move each vertex slightly away from the circumcenter direction
        for idx in worst_simplex:
            p = points[idx]
            # Move in direction that reduces the circumradius
            # This means moving the point so the triangle becomes more "flat"
            move_dir = -n_vec * np.dot(p, n_vec) + p * (1.0 - np.dot(p, n_vec))
            move_dir = np.cross(n_vec, p)  # Tangential direction
            move_dir /= (np.linalg.norm(move_dir) + 1e-15)
            
            new_p = p + step * move_dir
            new_p /= np.linalg.norm(new_p)
            points[idx] = new_p
        
        # Check improvement
        r = _covering_radius(points)
        if r < best_radius - 1e-10:
            best_radius = r
            best_points = points.copy()
            step *= 0.9
        else:
            # Revert
            for idx in worst_simplex:
                pass  # We already modified points, need to revert properly
            points = best_points.copy()
            step *= 0.8
            if step < 1e-6:
                break
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    
    # Start with Fibonacci spiral
    best_points = _fibonacci_start()
    best_radius = _covering_radius(best_points)
    
    # Multiple random restarts with hill climbing
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Generate a starting point
        if restart == 0:
            points = _fibonacci_start()
        elif restart % 3 == 0:
            points = _random_start()
        else:
            # Perturb the best found so far
            points = best_points + np.random.randn(N, 3) * 0.05
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Hill climb from this start
        remaining_time = DEADLINE - 10 - time.time()
        if remaining_time < 5:
            break
        
        # Allocate time based on restart number
        time_per_restart = remaining_time / (n_restarts - restart)
        time_per_restart = min(time_per_restart, 8.0)  # Cap at 8 seconds per restart
        time_per_restart = max(time_per_restart, 2.0)
        
        optimized_points, radius = _hill_climb(points, max_time=time_per_restart)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized_points
    
    # Final refinement with gradient approach
    remaining = DEADLINE - 5 - time.time()
    if remaining > 3:
        refined_points, radius = _refine_with_gradient(best_points, max_time=min(remaining, 10.0))
        if radius < best_radius:
            best_radius = radius
            best_points = refined_points
    
    # Ensure final points are valid
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / np.maximum(norms, 1e-15)
    
    return best_points
# EVOLVE-BLOCK-END
