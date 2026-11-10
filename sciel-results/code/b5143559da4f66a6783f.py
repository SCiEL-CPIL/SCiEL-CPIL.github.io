"""Combined program for covering the sphere with 130 equal caps."""
import numpy as np
import time
import math
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

N = 130
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START


def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    
    The covering radius equals the largest circumradius of the facets of the convex hull.
    Returns (radius_in_degrees, worst_center, worst_facet_idx, hull)
    """
    points = np.asarray(points, dtype=np.float64)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use nearest-neighbor based estimate
        tree = cKDTree(points)
        n_samples = 200000
        u = np.random.randn(n_samples, 3)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        _, idx = tree.query(u)
        dists = np.arccos(np.clip(np.sum(u * points[idx], axis=1), -1, 1))
        return np.degrees(np.max(dists)), None, -1, None
    
    max_radius = 0.0
    worst_center = None
    worst_facet_idx = -1
    
    for i, simplex in enumerate(hull.simplices):
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]
        
        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)
        
        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])
        
        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue
            
        xT_M_x = np.dot(x, M @ x)
        
        if xT_M_x <= 1e-10:
            continue
            
        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C
        
        R = math.acos(cos_R)
        
        if R > max_radius:
            max_radius = R
            worst_center = C
            worst_facet_idx = i
    
    return math.degrees(max_radius), worst_center, worst_facet_idx, hull


def _fast_covering_radius(points, test_points):
    """Faster covering radius estimate using sampling. Used during optimization."""
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    tree = cKDTree(points)
    _, idx = tree.query(test_points)
    dists = np.arccos(np.clip(np.sum(test_points * points[idx], axis=1), -1, 1))
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


def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards c by step_size radians."""
    dot_pc = np.dot(p, c)
    # Tangent vector from p to c
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p


def _hill_climb(points, max_time=None, test_points=None):
    """Perform hill-climbing optimization to minimize the covering radius."""
    points = points.copy()
    n = len(points)
    
    start_time = time.time()
    
    if test_points is None:
        n_samples = 50000
        test_points = np.random.randn(n_samples, 3)
        test_points /= np.linalg.norm(test_points, axis=1, keepdims=True)
    
    best_radius, _, _, _ = _covering_radius(points)
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
                r_fast = _fast_covering_radius(points, test_points)
                
                if r_fast < best_radius - 0.005:
                    # Full check
                    r_full, _, _, _ = _covering_radius(points)
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
    best_radius, _, _, _ = _covering_radius(points)
    best_points = points.copy()
    
    step = 0.01
    
    while True:
        if max_time is not None and time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE - 5:
            break
        
        # Compute covering radius and identify worst facet
        radius, center, facet_idx, hull = _covering_radius(points)
        
        if center is None or facet_idx == -1:
            break
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        # Strategy: Try moving the 3 vertices of the worst facet towards its circumcenter
        improved = False
        
        # Try moving all 3 vertices
        new_points = points.copy()
        new_points[v0] = _move_point_towards(points[v0], center, step)
        new_points[v1] = _move_point_towards(points[v1], center, step)
        new_points[v2] = _move_point_towards(points[v2], center, step)
        
        new_radius, _, _, _ = _covering_radius(new_points)
        
        if new_radius < radius:
            points = new_points
            improved = True
        else:
            # Try moving vertices individually
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                test_radius, _, _, _ = _covering_radius(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break
            
        if not improved:
            # Reduce step size
            step *= 0.8
            if step < 0.001:
                # Stuck. Perturb the system to escape local minimum
                noise = np.random.randn(3, 3) * 0.05
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]
                
                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                
                pert_radius, _, _, _ = _covering_radius(new_points)
                
                if pert_radius < best_radius * 1.05:
                    points = new_points
                    step = 0.05
                else:
                    step = 0.01
    
    return best_points, best_radius


def _global_repulsion(points, max_time=None):
    """Global repulsion phase to spread points out."""
    points = points.copy()
    start_time = time.time()
    
    best_radius, _, _, _ = _covering_radius(points)
    best_points = points.copy()
    
    alpha = 0.05
    for step in range(300):
        if max_time is not None and time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE - 5:
            break
            
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                # Repulsive force ~ 1/d^3
                force = diff / (dist_sq ** 1.5)
                forces[i] += force
                forces[j] -= force
        
        # Normalize forces to prevent large jumps
        forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / np.where(forces_norm == 0, 1, forces_norm)
        
        new_points = points + alpha * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        radius, _, _, _ = _covering_radius(new_points)
        if radius < best_radius:
            best_radius = radius
            best_points = new_points.copy()
            points = new_points
        else:
            # If repulsion makes it worse, reduce alpha
            alpha *= 0.95
            if alpha < 0.005:
                break
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    
    # Start with Fibonacci spiral
    best_points = _fibonacci_start()
    best_radius, _, _, _ = _covering_radius(best_points)
    
    # Precompute test points for fast evaluation
    n_samples = 50000
    test_points = np.random.randn(n_samples, 3)
    test_points /= np.linalg.norm(test_points, axis=1, keepdims=True)
    
    # Phase 1: Global Repulsion from Fibonacci start
    remaining_time = DEADLINE - 10 - time.time()
    if remaining_time > 5:
        rep_points, rep_radius = _global_repulsion(best_points, max_time=min(remaining_time, 15.0))
        if rep_radius < best_radius:
            best_radius = rep_radius
            best_points = rep_points
    
    # Multiple random restarts with hill climbing
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Generate a starting point
        if restart == 0:
            points = _fibonacci_start()
            # Add small jitter
            noise = np.random.randn(N, 3) * 0.01
            points = points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
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
        time_per_restart = min(time_per_restart, 5.0)  # Cap at 5 seconds per restart
        time_per_restart = max(time_per_restart, 1.5)
        
        optimized_points, radius = _hill_climb(points, max_time=time_per_restart, test_points=test_points)
        
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
