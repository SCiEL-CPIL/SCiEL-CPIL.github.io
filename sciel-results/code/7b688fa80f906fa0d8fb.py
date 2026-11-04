"""Rewritten program: Riesz s-energy + smooth LSE covering radius refinement via L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
WALL_CLOCK_LIMIT = 145.0  # safety margin below 150s


def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees using convex hull facets.
    """
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute-force on a fine grid
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2 * np.pi, 400)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        Y = np.column_stack([
            np.sin(TH) * np.cos(PH),
            np.sin(TH) * np.sin(PH),
            np.cos(TH)
        ]).reshape(-1, 3)
        Y = _normalize(Y)
        YP = Y @ points.T
        min_dist_sq = np.min(2.0 - 2.0 * YP, axis=1)
        max_dist = np.sqrt(np.max(min_dist_sq))
        return np.degrees(np.arccos(np.clip(1.0 - 0.5 * max_dist * max_dist, -1.0, 1.0)))
    
    # For each hull facet, compute the circumradius
    facets = hull.simplices
    max_ang = 0.0
    for tri in facets:
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        # Circumcenter of triangle on the unit sphere
        M = np.array([
            [a @ a, a @ b, a @ c],
            [b @ a, b @ b, b @ c],
            [c @ a, c @ b, c @ c]
        ])
        rhs = np.array([a @ a, b @ b, c @ c])
        try:
            coeffs = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        O = coeffs[0] * a + coeffs[1] * b + coeffs[2] * c
        O_norm = np.linalg.norm(O)
        if O_norm < 1e-15:
            continue
        O = O / O_norm
        # Check that O is on the unit sphere and the cap is empty
        cos_ang = np.clip(a @ O, -1.0, 1.0)
        ang = np.arccos(cos_ang)
        if ang > max_ang:
            max_ang = ang
    return np.degrees(max_ang)


def _smooth_covering_radius_and_grad(points, temperature=0.02):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    For each point y on the sphere (sampled), the distance to the nearest config point
    is approximated. We use a fine grid of sample points on the sphere.
    
    The smooth covering radius is:
    T * log(sum_y exp(d(y) / T))
    where d(y) is the angular distance from y to the nearest config point.
    
    Gradient w.r.t. config point p_i:
    For each sample y, let j(y) be the index of the nearest config point.
    d(y) = arccos(max_k y . p_k)
    The gradient of d(y) w.r.t. p_i is non-zero only if i = j(y).
    """
    # Sample points on the sphere (Fibonacci grid for good coverage)
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    Y = _normalize(Y)
    
    # Compute distances from each sample to each config point
    # dist_sq[y, i] = 2 - 2 * Y[y] . P[i]
    YP = Y @ points.T  # shape (n_samples, N)
    dist_sq = 2.0 - 2.0 * YP  # shape (n_samples, N)
    dist = np.sqrt(np.maximum(dist_sq, 0.0))  # chordal distance
    
    # Convert to angular distance
    cos_ang = np.clip(1.0 - 0.5 * dist_sq, -1.0, 1.0)
    ang_dist = np.arccos(cos_ang)  # shape (n_samples, N)
    
    # For each sample, find the minimum angular distance and which point achieves it
    min_ang = np.min(ang_dist, axis=1)  # shape (n_samples,)
    min_idx = np.argmin(ang_dist, axis=1)  # shape (n_samples,)
    
    # Smooth covering radius: T * log(sum exp(min_ang / T))
    # To avoid overflow, subtract max
    max_val = np.max(min_ang)
    log_sum = np.log(np.sum(np.exp((min_ang - max_val) / temperature))) + max_val / temperature
    smooth_cr = temperature * log_sum
    
    # Gradient: for each sample y, the gradient w.r.t. p_i is non-zero only if i == min_idx[y]
    # d/dp_i arccos(y . p_i) = -y / sqrt(1 - (y . p_i)^2)
    # But we need the gradient of min_k arccos(y . p_k) w.r.t. p_i
    # If i == min_idx[y], then d/dp_i arccos(y . p_i) = -y / sqrt(1 - (y . p_i)^2)
    
    # Weight each sample by exp(min_ang[y] / T) / sum_j exp(min_ang[j] / T)
    weights = np.exp((min_ang - max_val) / temperature)
    weights /= np.sum(weights)  # normalized weights
    
    # For each config point i, sum over samples where min_idx[y] == i
    grad = np.zeros((N, 3))
    for y_idx in range(n_samples):
        i_idx = min_idx[y_idx]
        y = Y[y_idx]
        p_i = points[i_idx]
        dot = y @ p_i
        denom = np.sqrt(max(1.0 - dot * dot, 1e-15))
        grad[i_idx] += weights[y_idx] * (-y / denom)
    
    # Scale by temperature (from the chain rule)
    grad *= temperature
    
    return smooth_cr, grad


def _riesz_energy_and_grad(points, s=12.0):
    """
    Compute the Riesz s-energy and its gradient for N points on the unit sphere.
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(n, dtype=bool)
    energy = np.sum(dist[mask] ** (-s))
    
    inv_dist_s = dist ** (-s)
    np.fill_diagonal(inv_dist_s, 0.0)
    inv_dist_s2 = inv_dist_s / dist_sq
    np.fill_diagonal(inv_dist_s2, 0.0)
    
    grad = -s * np.sum(inv_dist_s2[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _lbfgs_refine(points, objective_func, grad_func, maxiter=2000):
    """
    Run L-BFGS-B optimization with the given objective and gradient.
    """
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(N, 3)
        pts = _normalize(pts)
        val, grad = grad_func(pts)
        # Project gradient to tangent space
        grad_proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return val, grad_proj.flatten()
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    x_opt = result.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    return x_opt


def _spherical_lloyd_iteration(points, n_iters=20):
    """
    Lloyd's algorithm for sphere covering: move each point to the centroid of its Voronoi cell.
    """
    points = _normalize(points)
    for _ in range(n_iters):
        # Compute Voronoi cells using convex hull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find the facets that include it
        new_points = np.zeros_like(points)
        for i in range(N):
            # Find all facets that include point i
            facet_indices = np.where(np.any(hull.simplices == i, axis=1))[0]
            if len(facet_indices) == 0:
                new_points[i] = points[i]
                continue
            
            # Collect all vertices of these facets
            verts = hull.simplices[facet_indices].flatten()
            verts = np.unique(verts)
            # The Voronoi cell of point i is the intersection of half-spaces
            # For a simple approximation, use the average of the facet circumcenters
            # Or better: the centroid of the Voronoi cell
            
            # Simple approach: move towards the centroid of the neighboring points
            neighbors = points[verts]
            centroid = np.mean(neighbors, axis=0)
            if np.linalg.norm(centroid) > 1e-10:
                new_points[i] = centroid / np.linalg.norm(centroid)
            else:
                new_points[i] = points[i]
        
        points = _normalize(new_points)
    
    return points


def _fibonacci_points(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _jittered_fibonacci_points(n=N, seed=0, jitter=0.1):
    rng = np.random.default_rng(seed)
    pts = _fibonacci_points(n).copy()
    # Add small random perturbations
    noise = rng.normal(size=(n, 3)) * jitter
    pts += noise
    return _normalize(pts)


def _random_points(n=N, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + WALL_CLOCK_LIMIT
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    configs.append(_fibonacci_points())
    
    # 2. Jittered Fibonacci
    for seed in range(3):
        configs.append(_jittered_fibonacci_points(seed=seed, jitter=0.15))
    
    # 3. Random
    for seed in range(5):
        configs.append(_random_points(seed=seed))
    
    best_points = None
    best_radius = np.inf
    
    for idx, init in enumerate(configs):
        if time.time() > deadline - 25.0:
            break
        
        try:
            # Stage 1: Riesz energy minimization
            def riesz_obj(pts):
                return _riesz_energy_and_grad(pts)
            
            optimized = _lbfgs_refine(init, None, riesz_obj, maxiter=1000)
            
            # Stage 2: Smooth covering radius refinement
            def lse_obj(pts):
                return _smooth_covering_radius_and_grad(pts, temperature=0.02)
            
            optimized = _lbfgs_refine(optimized, None, lse_obj, maxiter=500)
            
            # Compute actual covering radius
            radius = _covering_radius(optimized)
            if radius < best_radius:
                best_radius = radius
                best_points = optimized.copy()
        except Exception:
            continue
    
    if best_points is None:
        return _fibonacci_points()
    
    # Final polish: run one more smooth refinement from the best point
    if time.time() < deadline - 15.0:
        try:
            def lse_obj_tight(pts):
                return _smooth_covering_radius_and_grad(pts, temperature=0.01)
            
            polished = _lbfgs_refine(best_points, None, lse_obj_tight, maxiter=1000)
            radius_polished = _covering_radius(polished)
            if radius_polished < best_radius:
                best_points = polished
        except Exception:
            pass
    
    return best_points
