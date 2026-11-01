import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50
DEADLINE = 112.0  # safety margin before 120s kill

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _covering_radius(points):
    """
    Compute the covering radius of the given points on the unit sphere.
    Returns the radius in radians, the worst simplex, and the worst circumcenter.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback if hull fails
        return np.pi, None, None

    worst_radius = 0
    worst_simplex = None
    worst_circumcenter = None

    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # Choose the normal pointing outward (away from origin relative to triangle centroid)
        # Actually, for covering radius, we need the circumcenter that is "far" from the origin?
        # No, the circumcenter is on the sphere.
        # The normal to the plane passing through a, b, c is n.
        # The circumcenter is either n or -n.
        # It is the one that has positive dot product with the centroid of the triangle?
        # Let's check: if a,b,c are close to each other, their centroid is close to n or -n.
        centroid = (a + b + c) / 3
        if np.dot(n, centroid) < 0:
            n = -n
            
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        
        if r > worst_radius:
            worst_radius = r
            worst_simplex = simplex
            worst_circumcenter = n

    return worst_radius, worst_simplex, worst_circumcenter

def _compute_facet_data(points):
    """
    Compute circumcenter and radius for all facets.
    Returns a list of tuples: (simplex, circumcenter, radius)
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return []

    facet_data = []
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        centroid = (a + b + c) / 3
        if np.dot(n, centroid) < 0:
            n = -n
            
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        facet_data.append((simplex, n, r))

    return facet_data

def _objective_mse(x_flat):
    """
    Objective: Sum of squared covering radii for all facets.
    This is smooth and helps establish a good global structure.
    """
    pts = x_flat.reshape(N, 3)
    # Normalize points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts / norms
    
    facet_data = _compute_facet_data(pts)
    if not facet_data:
        return 1e6
    
    radii = np.array([fd[2] for fd in facet_data])
    return np.sum(radii ** 2)

def _objective_max(x_flat):
    """
    Objective: Maximum covering radius.
    """
    pts = x_flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts / norms
    
    rad, _, _ = _covering_radius(pts)
    return rad

def _local_search(points, max_steps=100, step_size=0.05):
    """
    Local search to minimize the covering radius.
    Moves the vertices of the worst facet towards its circumcenter.
    """
    current_points = points.copy()
    best_radius, _, _ = _covering_radius(current_points)
    best_points = current_points.copy()
    
    for step in range(max_steps):
        radius, simplex, circumcenter = _covering_radius(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        if simplex is None or circumcenter is None:
            break
            
        if radius < 1e-6:
            break
            
        # Get the 3 points of the worst facet
        idx1, idx2, idx3 = simplex
        
        # Move these points towards the circumcenter
        alpha = step_size
        
        p1 = current_points[idx1]
        p2 = current_points[idx2]
        p3 = current_points[idx3]
        
        new_p1 = p1 + alpha * (circumcenter - p1)
        new_p2 = p2 + alpha * (circumcenter - p2)
        new_p3 = p3 + alpha * (circumcenter - p3)
        
        new_p1 = _normalize(new_p1)
        new_p2 = _normalize(new_p2)
        new_p3 = _normalize(new_p3)
        
        current_points[idx1] = new_p1
        current_points[idx2] = new_p2
        current_points[idx3] = new_p3
        
    return best_points, best_radius

def construct():
    start_time = time.time()
    
    best_points = None
    best_radius = np.inf
    
    # Phase 1: Global structure optimization using MSE objective
    # We use multiple restarts with L-BFGS-B on the MSE objective
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() - start_time > DEADLINE - 30: # Leave 30s for polish
            break
            
        # Initialize points
        rng = np.random.default_rng(seed=1000 + i)
        
        # Base Fibonacci
        j = np.arange(N) + 0.5
        z = 1.0 - 2.0 * j / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * j
        r = np.sqrt(1.0 - z * z)
        base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add noise
        noise = rng.normal(0, 0.1, size=(N, 3))
        points = base_points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        x0 = points.flatten()
        
        try:
            # Optimize MSE
            result = scipy.optimize.minimize(
                _objective_mse, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success:
                pts = result.x.reshape(N, 3)
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
                
                # Evaluate covering radius
                rad, _, _ = _covering_radius(pts)
                
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
                    
        except Exception:
            pass

    # Phase 2: Local polish using Max objective
    if best_points is not None:
        # Run local search to refine the max radius
        pts, rad = _local_search(best_points, max_steps=200, step_size=0.05)
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
            
        # Further polish with L-BFGS-B on Max objective (might be less smooth but targeted)
        # We can try this if we have time
        if time.time() - start_time < DEADLINE - 10:
            x0 = best_points.flatten()
            try:
                result = scipy.optimize.minimize(
                    _objective_max, 
                    x0, 
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success and result.fun < best_radius:
                    best_radius = result.fun
                    best_points = result.x.reshape(N, 3)
                    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
            except Exception:
                pass
        
        # Final local search
        if time.time() - start_time < DEADLINE - 2:
            pts, rad = _local_search(best_points, max_steps=100, step_size=0.01)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()

    if best_points is None:
        # Fallback to seed
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_points
