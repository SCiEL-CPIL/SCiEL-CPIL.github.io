"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    Uses convex hull facet circumradii.
    """
    if len(points) < 3:
        return 180.0
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
    
    max_angle = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        if angle > max_angle:
            max_angle = angle
            
    return max_angle * 180.0 / np.pi


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _generate_test_grid(n_theta=30, n_phi=60):
    """Generate a grid of test points on the sphere."""
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


def _facet_circumradius_angles(points):
    """Compute circumradius angle for each facet of the convex hull."""
    try:
        hull = ConvexHull(points)
    except:
        return np.array([np.pi/2])  # Fallback
    
    angles = np.zeros(len(hull.simplices))
    for i, simplex in enumerate(hull.simplices):
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            angles[i] = np.pi/2
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angles[i] = np.arccos(h)
    
    return angles


def _facet_smooth_surrogate(params, beta=50.0):
    """
    Smooth surrogate based on convex hull facet circumradii.
    For each facet, compute the circumradius angle and use soft-max.
    This directly targets the true covering radius.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    angles = _facet_circumradius_angles(pts)
    
    # Soft-max over facet angles: logsumexp(beta * angles) / beta
    beta_angles = beta * angles
    max_val = np.max(beta_angles)
    logsumexp_max = max_val + np.log(np.sum(np.exp(beta_angles - max_val)))
    soft_max = logsumexp_max / beta
    
    return soft_max


def _smooth_gradient_optimize(points, time_limit=30, beta=100.0, use_facet=True):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    Can use either grid-based or facet-based surrogate.
    """
    start_time = time.time()
    
    if use_facet:
        def objective(params):
            if time.time() - start_time > time_limit:
                return 1e6
            return _facet_smooth_surrogate(params, beta=beta)
    else:
        grid = _generate_test_grid(n_theta=25, n_phi=50)
        def objective(params):
            if time.time() - start_time > time_limit:
                return 1e6
            return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(N, 3)
    return _normalize(points)


def _icosahedron_based_points():
    """Generate points based on icosahedron vertices with perturbations."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    rng = np.random.RandomState(123)
    points = [verts[i] for i in range(12)]
    
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.1:
            points.append(new_pt)
    
    return np.array(points[:N])


def _spherical_lloyd_iteration(points, max_iter=100, time_limit=30, damping=0.5):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves points towards the circumcenter of their largest adjacent facet.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        new_points = points.copy()
        
        # Find the largest facet for each point
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        # Compute facet circumradius angles
        facet_angles = _facet_circumradius_angles(points)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            # Find the largest facet adjacent to this point
            max_angle_idx = max(facets, key=lambda fi: facet_angles[fi])
            simplex = hull.simplices[max_angle_idx]
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            if np.dot(n, p1) < 0:
                n = -n
            
            # Move towards the circumcenter (normal direction)
            new_points[pi] = points[pi] + damping * (n - points[pi])
            new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            # Try with smaller damping
            new_points2 = points.copy()
            for pi in range(N):
                facets = point_to_facets[pi]
                if not facets:
                    continue
                max_angle_idx = max(facets, key=lambda fi: facet_angles[fi])
                simplex = hull.simplices[max_angle_idx]
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                if np.dot(n, p1) < 0:
                    n = -n
                
                new_points2[pi] = points[pi] + damping * 0.3 * (n - points[pi])
                new_points2[pi] = new_points2[pi] / np.linalg.norm(new_points2[pi])
            
            new_points2 = _normalize(new_points2)
            new_cost2 = _covering_radius(new_points2)
            
            if new_cost2 < current_cost:
                current_cost = new_cost2
                points = new_points2
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_points = points.copy()
            else:
                break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=20, maxiter=2000):
    """
    Local optimization using Nelder-Mead on the true covering radius.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius_radians(pts)
    
    x0 = points.flatten()
    
    # Use a smaller initial simplex for better convergence
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-16, 
                               'adaptive': True, 'initial_simplex': None,
                               'maxls': 50})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 115.0  # Safety margin for 120s limit
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Random with different seeds
    for seed in [42, 123, 777, 999]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Lloyd's relaxation from multiple starts
    # Allocate ~25s for this phase across all initializations
    lloyd_time_each = 5.0
    for init_pts in initializations:
        if _time_left() < 35: # Keep reserve for later phases
            break
        
        pts = _spherical_lloyd_iteration(init_pts, max_iter=150, time_limit=min(lloyd_time_each, _time_left() - 30), damping=0.5)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Smooth gradient optimization using facet-based surrogate
    # This directly targets the true covering radius
    if _time_left() > 45:
        # Run facet-based smooth optimization with moderate beta
        pts = _smooth_gradient_optimize(best_points, time_limit=min(25, _time_left() - 35), beta=60.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
        # Run again with higher beta for sharper gradient
        if _time_left() > 30:
            pts = _smooth_gradient_optimize(best_points, time_limit=min(15, _time_left() - 25), beta=120.0, use_facet=True)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Phase 3: Final High-Precision Polish
    # Use Nelder-Mead on the exact objective for final tightening
    if _time_left() > 20:
        pts = _nelder_mead_optimize(best_points, time_limit=min(15, _time_left() - 15), maxiter=5000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
    # Final safety check
    if _time_left() > 8:
        # One last quick Nelder-Mead with very tight tolerance
        pts = _nelder_mead_optimize(best_points, time_limit=min(5, _time_left() - 3), maxiter=2000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
