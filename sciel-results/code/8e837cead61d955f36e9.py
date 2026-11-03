"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    Uses convex hull facet circumradii (vectorized).
    """
    if len(points) < 3:
        return 180.0
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
    
    # Vectorized computation of all facet normals and heights
    simplices = hull.simplices  # shape (n_facets, 3)
    p1 = points[simplices[:, 0]]  # (n_facets, 3)
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)  # (n_facets, 3)
    norm_n = np.linalg.norm(normals, axis=1, keepdims=True)  # (n_facets, 1)
    
    # Handle degenerate facets
    valid = norm_n.squeeze() > 1e-10
    if not np.any(valid):
        return 180.0
    
    normals = normals / norm_n  # (n_facets, 3)
    
    # Ensure normals point outward
    dots = np.einsum('ij,ij->i', normals, p1)
    normals = np.where(dots[:, np.newaxis] < 0, -normals, normals)
    
    # Compute h = dot(normal, p1) for each facet
    h = np.einsum('ij,ij->i', normals, p1)
    h = np.clip(h, -1.0, 1.0)
    
    # Covering radius is max arccos(h) = arccos(min(h))
    min_h = np.min(h)
    angle = np.arccos(min_h)
    
    return float(angle * 180.0 / np.pi)


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _smooth_surrogate(params, beta=50.0):
    """
    Smooth surrogate using log-sum-exp approximation of max.
    Minimizes a smooth upper bound on the covering radius.
    Vectorized for speed.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except:
        return 100.0
    
    # Vectorized computation of all facet normals and heights
    simplices = hull.simplices
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norm_n = np.linalg.norm(normals, axis=1, keepdims=True)
    
    valid = norm_n.squeeze() > 1e-10
    if not np.any(valid):
        return 100.0
    
    normals = normals / norm_n
    dots = np.einsum('ij,ij->i', normals, p1)
    normals = np.where(dots[:, np.newaxis] < 0, -normals, normals)
    
    h = np.einsum('ij,ij->i', normals, p1)
    h = np.clip(h, -1.0, 1.0)
    
    if len(h) == 0:
        return 100.0
    
    # Minimize -min(h) smoothly: use soft-min via logsumexp
    # soft_min(h) = -logsumexp(-beta*h)/beta
    # We want to minimize -soft_min(h) = logsumexp(-beta*h)/beta
    neg_beta_h = -beta * h
    max_val = np.max(neg_beta_h)
    logsumexp = max_val + np.log(np.sum(np.exp(neg_beta_h - max_val)))
    
    return logsumexp / beta


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
    # 12 vertices of regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # Expand to 50 points by adding subdivisions
    rng = np.random.RandomState(123)
    points = [verts[i] for i in range(12)]
    
    # Add 38 more points by perturbing and adding
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        # Check minimum distance to existing points
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.1:
            points.append(new_pt)
    
    return np.array(points[:N])


def _spherical_lloyd_iteration(points, max_iter=200, time_limit=60, damping=0.5):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Vectorized facet normal computation for speed.
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
        
        # Pre-compute all facet normals vectorized
        simplices = hull.simplices
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normals = np.cross(v1, v2)
        norm_n = np.linalg.norm(normals, axis=1, keepdims=True)
        
        valid = norm_n.squeeze() > 1e-10
        normals = normals / norm_n
        dots = np.einsum('ij,ij->i', normals, p1)
        normals = np.where(dots[:, np.newaxis] < 0, -normals, normals)
        
        # For each point, sum normals of adjacent facets
        normal_sums = np.zeros((N, 3))
        for fi, simplex in enumerate(simplices):
            if not valid[fi]:
                continue
            for pi in simplex:
                normal_sums[pi] += normals[fi]
        
        new_points = points.copy()
        for pi in range(N):
            ns = normal_sums[pi]
            ns_norm = np.linalg.norm(ns)
            if ns_norm > 1e-10:
                target = ns / ns_norm
                new_points[pi] = points[pi] + damping * (target - points[pi])
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
            break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=20):
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
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _smooth_gradient_optimize(points, time_limit=20, beta=50.0):
    """
    Gradient-based optimization using smooth surrogate.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _sa_optimize_covering(points, T_start=1.0, T_end=1e-4, n_iter=5000, time_limit=100):
    """
    Simulated Annealing to minimize covering radius directly.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        T = T_start * (cooling ** i)
        
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        step_size = 0.05 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        new_cost = _covering_radius(points)
        delta_C = new_cost - current_cost
        
        if delta_C < 0 or np.random.rand() < np.exp(-delta_C / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            points[idx] = original_point
            
    return best_points


def _refine_point(points, idx, time_limit=5):
    """
    Refine a single point by optimizing its position given others fixed.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    other_points = np.delete(points, idx, axis=0)
    
    def objective_single(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pt = params / np.linalg.norm(params)
        all_pts = np.vstack([other_points, pt.reshape(1, 3)])
        return _covering_radius_radians(all_pts)
    
    x0 = points[idx]
    result = minimize(objective_single, x0, method='Nelder-Mead',
                      options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
    new_pt = result.x / np.linalg.norm(result.x)
    points[idx] = new_pt
    return points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Random with different seeds
    for seed in [42, 123, 777]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Multi-start with Lloyd relaxation
    for init_pts in initializations:
        if _time_left() < 25:
            break
        
        # Lloyd's relaxation with more iterations thanks to vectorized cost
        pts = _spherical_lloyd_iteration(init_pts, max_iter=80, time_limit=min(12, _time_left() - 20), damping=0.5)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Smooth gradient optimization with adaptive beta
    if _time_left() > 25:
        # Start with higher beta for sharper approximation
        pts = _smooth_gradient_optimize(best_points, time_limit=min(12, _time_left() - 20), beta=80.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
        
        # Second pass with lower beta for smoother landscape
        if _time_left() > 20:
            pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 15), beta=30.0)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Phase 3: Nelder-Mead on true objective
    if _time_left() > 20:
        pts = _nelder_mead_optimize(best_points, time_limit=min(12, _time_left() - 15))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Simulated Annealing
    if _time_left() > 25:
        pts = _sa_optimize_covering(best_points, T_start=0.3, T_end=1e-4, n_iter=5000, time_limit=min(15, _time_left() - 12))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5: Final Lloyd's polish with tighter damping
    if _time_left() > 12:
        pts = _spherical_lloyd_iteration(best_points, max_iter=50, time_limit=min(6, _time_left() - 8), damping=0.2)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: Final Nelder-Mead polish
    if _time_left() > 10:
        pts = _nelder_mead_optimize(best_points, time_limit=min(6, _time_left() - 6))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 7: Per-point refinement
    if _time_left() > 15:
        for idx in range(N):
            if _time_left() < 10:
                break
            pts = _refine_point(best_points, idx, time_limit=min(2, _time_left() - 8))
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Phase 8: Final high-precision polish
    if _time_left() > 8:
        pts = _spherical_lloyd_iteration(best_points, max_iter=30, time_limit=min(3, _time_left() - 5), damping=0.1)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
