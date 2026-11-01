import numpy as np
import scipy.spatial
import time

N = 50

def _covering_radius(points):
    """
    Compute the covering radius of the given points on the unit sphere.
    Returns the radius in radians, the worst simplex, and the worst circumcenter.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based estimate
        grid = _generate_grid(80)
        min_dists = np.min(np.arccos(np.clip(grid @ points.T, -1, 1)), axis=1)
        return np.max(min_dists), np.array([0, 1, 2]), points[0]

    worst_radius = 0.0
    worst_simplex = None
    worst_circumcenter = None

    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
        
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        
        if r > worst_radius:
            worst_radius = r
            worst_simplex = simplex
            worst_circumcenter = n

    if worst_simplex is None:
        return 0.0, np.array([0, 1, 2]), points[0]
    
    return worst_radius, worst_simplex, worst_circumcenter

def _generate_grid(n_theta=50, n_phi=100):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _compute_all_circumcenters(points):
    """Compute circumcenters for all convex hull facets."""
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return []
    
    circumcenters = []
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
        
        circumcenters.append(n)
    
    return np.array(circumcenters) if circumcenters else np.array([])

def _repulsive_force_optimize(points, max_iters=300, time_limit=110.0, start_time=None):
    """
    Optimize using repulsive forces between points and pressure from circumcenters.
    The energy combines:
    1. Repulsion between all pairs (inverse square of angular distance)
    2. Pressure from circumcenters (push points away from worst circumcenters)
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    best = current.copy()
    best_radius, _, _ = _covering_radius(current)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all circumcenters and their distances
        circumcenters = _compute_all_circumcenters(current)
        
        if len(circumcenters) == 0:
            break
        
        # Find the worst circumcenter (max distance to its facet vertices)
        # and compute pressure forces
        forces = np.zeros_like(current)
        
        # Pairwise repulsion: points repel each other
        # F_ij = (1/d_ij^2) * (p_i - p_j) normalized to sphere
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = np.arccos(np.clip(np.dot(current[i], current[j]), -1, 1))
                if d < 1e-6:
                    continue
                # Inverse square repulsion
                f = (1.0 / (d * d)) * (current[i] - current[j])
                forces[i] += f
        
        # Pressure from circumcenters: push points away from circumcenters
        # that are far from the nearest point
        for cc in circumcenters:
            # Distance from circumcenter to each point
            dists = np.arccos(np.clip(current @ cc, -1, 1))
            min_dist = np.min(dists)
            
            # Only apply pressure if this circumcenter is a "worst" point
            # (i.e., its distance to nearest point is significant)
            if min_dist > 0.1:  # ~5.7 degrees
                # Push the closest points away from this circumcenter
                for i in range(N):
                    d = dists[i]
                    if d < min_dist * 1.5:  # Only nearby points
                        # Force away from circumcenter
                        f = (1.0 / (d + 0.1)) * (current[i] - cc)
                        forces[i] += 0.5 * f
        
        # Normalize forces to be tangent to sphere
        for i in range(N):
            # Project force onto tangent plane
            radial = np.dot(forces[i], current[i]) * current[i]
            forces[i] = forces[i] - radial
        
        # Adaptive step size
        base_step = 0.02 * (1.0 - it / max_iters)
        
        # Apply forces
        new_points = current.copy()
        for i in range(N):
            step = base_step * forces[i]
            new_points[i] = current[i] + step
            # Normalize to sphere
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-10:
                new_points[i] = new_points[i] / norm
        
        # Check if improvement
        new_radius, _, _ = _covering_radius(new_points)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best = new_points.copy()
            current = new_points
        else:
            # If no improvement, try smaller step or accept with probability
            # (simulated annealing-like)
            if np.random.random() < 0.1 and new_radius < best_radius * 1.05:
                current = new_points
            else:
                # Keep current, but maybe add small random perturbation
                if it % 50 == 0:
                    for i in range(N):
                        if np.random.random() < 0.3:
                            pert = np.random.normal(0, 0.01, 3)
                            new_p = current[i] + pert
                            new_p = new_p / np.linalg.norm(new_p)
                            current[i] = new_p
    
    return best, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin before 150s kill
    
    best_points = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 15
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(seed=1000 + i * 7)
        
        # Different initialization strategies
        if i < 5:
            # Fibonacci with varying noise
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(1.0 - z * z)
            base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            noise = rng.normal(0, 0.05 + 0.05 * i, size=(N, 3))
            points = base_points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif i < 10:
            # Random uniform on sphere
            points = rng.normal(0, 1, size=(N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Fibonacci with larger perturbation
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(1.0 - z * z)
            base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            noise = rng.normal(0, 0.2, size=(N, 3))
            points = base_points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Run repulsive force optimization
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        pts, rad = _repulsive_force_optimize(
            points, 
            max_iters=200, 
            time_limit=remaining - 2.0,
            start_time=time.time()
        )
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts
        
        # Early termination if we're close to known best
        if best_radius < 0.319:  # ~18.3 degrees
            # Try a few more iterations with the best points
            remaining = deadline - time.time()
            if remaining > 10:
                pts, rad = _repulsive_force_optimize(
                    best_points,
                    max_iters=100,
                    time_limit=remaining - 2.0,
                    start_time=time.time()
                )
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts
    
    # Final polish
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 5:
            pts, rad = _repulsive_force_optimize(
                best_points,
                max_iters=50,
                time_limit=remaining - 1.0,
                start_time=time.time()
            )
            if rad < best_radius:
                best_radius = rad
                best_points = pts
    
    if best_points is None:
        # Fallback: Fibonacci sphere
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure all points are unit vectors
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
