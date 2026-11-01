import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50

def _covering_radius(points):
    """
    Compute the covering radius of the given points on the unit sphere.
    Returns the radius in radians.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based estimate
        n_theta, n_phi = 60, 120
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        min_dists = np.min(np.arccos(np.clip(grid @ points.T, -1, 1)), axis=1)
        return np.max(min_dists)

    worst_radius = 0.0
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
    
    return worst_radius

def _get_worst_facet(points):
    """
    Get the worst facet (largest circumradius) and its circumcenter.
    Returns (radius, circumcenter, vertex_indices) or (inf, None, None) on failure.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return np.inf, None, None

    worst_radius = 0.0
    worst_center = None
    worst_vertices = None

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
            worst_center = n
            worst_vertices = simplex.copy()
    
    if worst_center is None:
        return np.inf, None, None
    
    return worst_radius, worst_center, worst_vertices

def _local_search(points, time_limit=20.0, start_time=None):
    """
    Local search using gradient-based optimization on the sphere.
    Uses a soft-max approximation of the max function for differentiability.
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    best = current.copy()
    best_radius = _covering_radius(current)
    
    # Use a dense grid to approximate the covering radius
    n_theta, n_phi = 40, 80
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Soft-max parameter
    beta = 50.0
    
    def objective(flat_params):
        # Convert flat parameters to points on sphere
        pts = flat_params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Compute distances from grid points to nearest site
        dots = grid @ pts.T  # shape (n_grid, N)
        dots = np.clip(dots, -1, 1)
        angles = np.arccos(dots)
        
        # For each grid point, find the nearest site
        min_dists = np.min(angles, axis=1)
        
        # Soft max of min_dists
        max_soft = -np.log(np.sum(np.exp(-beta * min_dists))) / beta
        
        return max_soft
    
    # Run L-BFGS-B optimization
    x0 = current.ravel()
    
    for restart in range(3):
        if time.time() - start_time > time_limit:
            break
        
        # Add small noise for restart
        if restart > 0:
            noise = np.random.normal(0, 0.01, size=(N, 3))
            x0 = (current + noise).ravel()
            norms = np.linalg.norm(x0.reshape(N, 3), axis=1, keepdims=True)
            x0 = (x0.reshape(N, 3) / norms).ravel()
        
        try:
            result = scipy.optimize.minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 5000}
            )
            
            if result.success or result.fun < best_radius:
                new_pts = result.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_pts = new_pts / norms
                
                new_radius = _covering_radius(new_pts)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best = new_pts.copy()
                    current = new_pts.copy()
        except Exception:
            pass
    
    return best, best_radius

def _lloyd_iteration(points):
    """
    One iteration of Lloyd's algorithm for spherical k-means.
    """
    # Assign each point on a grid to its nearest site
    n_theta, n_phi = 50, 100
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Find nearest site for each grid point
    dots = grid @ points.T
    dots = np.clip(dots, -1, 1)
    nearest = np.argmax(dots, axis=1)
    
    # Compute new centroids as normalized sum of assigned grid points
    new_points = np.zeros_like(points)
    for i in range(N):
        mask = nearest == i
        if np.sum(mask) > 0:
            new_points[i] = np.sum(grid[mask], axis=0)
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-10:
                new_points[i] = new_points[i] / norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
    
    return new_points

def _repulsion_relax(points, max_iters=50, time_limit=10.0, start_time=None):
    """
    Relax points using repulsive forces to spread them out.
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = np.arccos(np.clip(np.dot(current[i], current[j]), -1, 1))
                if d < 1e-6:
                    continue
                f = (1.0 / (d * d)) * (current[i] - current[j])
                forces[i] += f
        
        # Normalize forces to tangent plane
        for i in range(N):
            radial = np.dot(forces[i], current[i]) * current[i]
            forces[i] = forces[i] - radial
        
        # Apply forces with adaptive step
        step = 0.01 * (1.0 - it / max_iters)
        new_points = current.copy()
        for i in range(N):
            new_points[i] = current[i] + step * forces[i]
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-10:
                new_points[i] = new_points[i] / norm
        
        current = new_points
    
    return current

def _worst_facet_step(points, step_size=0.05, time_limit=10.0, start_time=None):
    """
    Move the vertices of the worst facet toward its circumcenter to reduce the covering radius.
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    best = current.copy()
    best_radius = _covering_radius(current)
    
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        
        radius, worst_center, worst_vertices = _get_worst_facet(current)
        if worst_vertices is None:
            break
        
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        
        # Move the three vertices of the worst facet toward the circumcenter
        for idx in worst_vertices:
            # Project the direction (worst_center - current[idx]) onto the tangent plane
            direction = worst_center - current[idx]
            # Remove radial component
            radial = np.dot(direction, current[idx]) * current[idx]
            tangent = direction - radial
            norm_tangent = np.linalg.norm(tangent)
            if norm_tangent < 1e-10:
                continue
            
            # Move toward circumcenter
            new_point = current[idx] + step_size * tangent / norm_tangent
            norm_new = np.linalg.norm(new_point)
            if norm_new > 1e-10:
                new_point = new_point / norm_new
            current[idx] = new_point
        
        # Slightly move other points to maintain spread
        for i in range(N):
            if i not in worst_vertices:
                # Small repulsion from worst facet vertices
                for j in worst_vertices:
                    d = np.arccos(np.clip(np.dot(current[i], current[j]), -1, 1))
                    if d < 0.3:  # Only repel if close
                        direction = current[i] - current[j]
                        radial = np.dot(direction, current[i]) * current[i]
                        tangent = direction - radial
                        norm_tangent = np.linalg.norm(tangent)
                        if norm_tangent > 1e-10:
                            current[i] = current[i] + 0.01 * tangent / norm_tangent
                            norm_new = np.linalg.norm(current[i])
                            if norm_new > 1e-10:
                                current[i] = current[i] / norm_new
    
    return best, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        rng = np.random.default_rng(seed=2000 + i * 13)
        
        # Different initialization strategies
        if i % 4 == 0:
            # Fibonacci with varying noise
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(1.0 - z * z)
            base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            noise = rng.normal(0, 0.03 + 0.02 * (i % 4), size=(N, 3))
            points = base_points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif i % 4 == 1:
            # Random uniform on sphere
            points = rng.normal(0, 1, size=(N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif i % 4 == 2:
            # Fibonacci with larger perturbation
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(1.0 - z * z)
            base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            noise = rng.normal(0, 0.1, size=(N, 3))
            points = base_points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Random with slight structure
            points = rng.normal(0, 1, size=(N, 3))
            # Add some correlation to create structure
            for k in range(N):
                if k > 0 and rng.random() < 0.3:
                    points[k] = 0.7 * points[k] + 0.3 * points[k-1]
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Step 1: Repulsion relaxation for initial spread
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        points = _repulsion_relax(points, max_iters=30, time_limit=min(8.0, remaining - 15), start_time=time.time())
        
        # Step 2: Lloyd's iterations
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            points = _lloyd_iteration(points)
        
        # Step 3: Worst facet gradient step
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        pts, rad = _worst_facet_step(points, step_size=0.05, time_limit=min(15.0, remaining - 10), start_time=time.time())
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts
        
        # Step 4: Local search with gradient-based optimization
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        pts, rad = _local_search(points, time_limit=min(25.0, remaining - 5), start_time=time.time())
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts
        
        # Early termination if we're close to known best
        if best_radius < 0.319:  # ~18.3 degrees
            remaining = deadline - time.time()
            if remaining > 10:
                pts, rad = _local_search(best_points, time_limit=min(15.0, remaining - 5), start_time=time.time())
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts
    
    # Final polish
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 5:
            pts, rad = _local_search(best_points, time_limit=min(10.0, remaining - 2), start_time=time.time())
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
