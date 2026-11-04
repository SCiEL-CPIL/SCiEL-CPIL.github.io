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
    Uses convex hull facet circumradii (vectorized).
    """
    if len(points) < 3:
        return 180.0
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
    
    simplices = hull.simplices
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    mask = norm_n[:, 0] > 1e-10
    if not np.any(mask):
        return 180.0
    
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    # Ensure normals point outward
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    h = np.sum(n * p1m, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    return float(np.max(angles)) * 180.0 / np.pi


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _generate_test_grid(n_theta=50, n_phi=100):
    """
    Generate a dense grid of test points on the sphere.
    Higher resolution leads to a tighter surrogate.
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


def _smooth_surrogate(params, beta=200.0, grid=None):
    """
    Vectorized smooth differentiable surrogate for the covering radius.
    Uses a grid of test points and log-sum-exp for soft min/max.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    if grid is None:
        grid = _generate_test_grid()
        
    # Compute angular distances: grid (M,3) x pts (N,3) -> (M,N)
    dots = np.clip(grid @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min over points for each grid point
    neg_beta_angles = -beta * angles
    max_neg = np.max(neg_beta_angles, axis=1, keepdims=True)
    exp_terms = np.exp(neg_beta_angles - max_neg)
    sum_exp = np.sum(exp_terms, axis=1)
    logsumexp_neg = max_neg.squeeze() + np.log(sum_exp)
    soft_min_angles = -logsumexp_neg / beta  # (M,)
    
    # Soft-max over grid points
    beta_soft_min = beta * soft_min_angles
    max_val = np.max(beta_soft_min)
    logsumexp_max = max_val + np.log(np.sum(np.exp(beta_soft_min - max_val)))
    soft_max = logsumexp_max / beta
    
    return soft_max


def _facet_smooth_surrogate(params, beta=50.0):
    """
    Smooth surrogate based on convex hull facet circumradii (vectorized).
    This provides a gradient for L-BFGS-B.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
        
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi

    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    # Normal to the plane
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    # Handle degenerate triangles
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return np.pi
        
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    # Ensure normals point towards the interior of the sphere relative to the triangle?
    # The circumcenter c is parallel to n.
    # We need c.p1 > 0.
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    # Calculate circumradius
    # cos(R) = c.p1
    cos_r = np.sum(n * p1m, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    # Soft-max (log-sum-exp)
    # We want to minimize the max radius, so we approximate max(radii) with logsumexp(beta * radii) / beta
    max_val = np.max(beta * radii)
    logsumexp = max_val + np.log(np.sum(np.exp(beta * radii - max_val)))
    return logsumexp / beta


def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _jittered_fibonacci_points(seed=123, jitter=0.05):
    """Generate Fibonacci points with small random jitter to break symmetries."""
    pts = _fibonacci_points()
    rng = np.random.RandomState(seed)
    noise = rng.randn(N, 3) * jitter
    pts = pts + noise
    return _normalize(pts)


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


def _spherical_lloyd_iteration(points, max_iter=50, time_limit=10, damping=0.3):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) with adaptive damping.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    current_damping = damping
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        simplices = hull.simplices
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1, keepdims=True)
        mask = norm_n[:, 0] > 1e-10
        if not np.any(mask):
            break
        
        n_valid = n[mask] / norm_n[mask]
        p1_valid = p1[mask]
        
        dots = np.sum(n_valid * p1_valid, axis=1)
        n_valid[dots < 0] *= -1
        
        h = np.sum(n_valid * p1_valid, axis=1)
        h = np.clip(h, -1.0, 1.0)
        facet_angles = np.arccos(h)
        
        # Accumulate targets for each point
        acc = np.zeros((N, 3))
        weights = np.zeros(N)
        
        simplices_valid = simplices[mask]
        for fi in range(len(simplices_valid)):
            angle = facet_angles[fi]
            n_vec = n_valid[fi]
            idxs = simplices_valid[fi]
            acc[idxs] += angle * n_vec
            weights[idxs] += angle
        
        new_points = points.copy()
        for pi in range(N):
            if weights[pi] > 1e-10:
                target = acc[pi] / np.linalg.norm(acc[pi])
                candidate = points[pi] + current_damping * (target - points[pi])
                candidate = candidate / np.linalg.norm(candidate)
                new_points[pi] = candidate
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
            current_damping = min(0.8, current_damping * 1.1)
        else:
            current_damping *= 0.5
            if current_damping < 0.01:
                break
    
    return best_points


def _smooth_gradient_optimize(points, time_limit=15, beta=200.0, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    if grid is None:
        grid = _generate_test_grid(n_theta=50, n_phi=100)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _facet_gradient_optimize(points, time_limit=15, beta=50.0):
    """
    Gradient-based optimization using facet smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _facet_smooth_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _nelder_mead_optimize(points, time_limit=10, maxiter=500):
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
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _block_coordinate_descent(points, time_limit=10, max_rounds=3):
    """
    Block coordinate descent: optimize each point individually while keeping others fixed.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        for idx in range(N):
            if time.time() - start_time > time_limit:
                break
            
            other_points = np.delete(points, idx, axis=0)
            
            def objective_single(params):
                pt = params / np.linalg.norm(params)
                all_pts = np.vstack([other_points, pt.reshape(1, 3)])
                return _covering_radius_radians(all_pts)
            
            x0 = points[idx]
            result = minimize(objective_single, x0, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            new_pt = result.x / np.linalg.norm(result.x)
            points[idx] = new_pt
        
        new_cost = _covering_radius(points)
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = points.copy()
    
    return best_points


def _repulsion_energy(points, exponent=6.0):
    """Compute repulsion energy to encourage spread-out points."""
    n = len(points)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-10:
                energy += 1e10
            else:
                energy += 1.0 / (d ** exponent)
    return energy


def _repulsion_gradient(points, exponent=6.0):
    """Compute gradient of repulsion energy."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            # d/dx_i (1/|x_i - x_j|^exponent) = -exponent * diff / |x_i - x_j|^(exponent+2)
            coeff = -exponent / (d ** (exponent + 2))
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
    return grad


def _repulsion_refine(points, time_limit=15.0, exponent=6.0):
    """Refine points using repulsion energy to get better initial spread."""
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10
        pts = _normalize(params.reshape(N, 3))
        return _repulsion_energy(pts, exponent)
    
    def gradient(params):
        pts = _normalize(params.reshape(N, 3))
        g = _repulsion_gradient(pts, exponent)
        # Project gradient to tangent space
        for i in range(N):
            g[i] -= np.dot(g[i], pts[i]) * pts[i]
        return g.flatten()
    
    x0 = points.flatten()
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 100})
        return _normalize(result.x.reshape(N, 3))
    except Exception:
        return points


def _multi_beta_lse_refine(points, time_budget=60.0, betas=[10, 50, 200, 500]):
    """Multi-stage beta cascade for LSE refinement."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5.0:
            break
        points = _facet_gradient_optimize(points, time_limit=remaining * 0.8, beta=beta)
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Precompute grid for smooth surrogate (shared across optimizations)
    grid = _generate_test_grid(n_theta=50, n_phi=100)
    
    # Generate diverse initializations
    initializations = []
    initializations.append(_fibonacci_points())
    
    # Jittered Fibonacci
    for seed in [123, 456, 789, 1011, 2020]:
        initializations.append(_jittered_fibonacci_points(seed=seed, jitter=0.05 + 0.01*seed))
    
    # Random
    for seed in [42, 123, 777, 999, 1234, 5555]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron
    initializations.append(_icosahedron_based_points())
    
    # Quick evaluation and ranking
    scored_inits = []
    for init_pts in initializations:
        cost = _covering_radius(init_pts)
        scored_inits.append((cost, init_pts))
    
    # Sort by cost (best first)
    scored_inits.sort(key=lambda x: x[0])
    
    # Focus on the top 3 candidates to save time
    top_candidates = [pts for _, pts in scored_inits[:3]]
    
    best_points = None
    best_cost = np.inf
    
    # Refine each top candidate
    for candidate in top_candidates:
        if _time_left() < 40:
            break
        
        pts = candidate.copy()
        
        # Stage 1: Repulsion refinement to improve initial spread
        if _time_left() > 50:
            pts = _repulsion_refine(pts, time_limit=min(5, _time_left() - 45))
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
        
        # Stage 2: Lloyd's Relaxation
        pts = _spherical_lloyd_iteration(pts, max_iter=30, time_limit=min(8, _time_left() - 35), damping=0.3)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
        
        # Stage 3: Multi-stage beta cascade refinement
        if _time_left() > 25:
            pts = _multi_beta_lse_refine(pts, time_budget=min(20, _time_left() - 20))
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
        
        # Stage 4: Exact Local Search (Nelder-Mead)
        if _time_left() > 10:
            pts = _nelder_mead_optimize(pts, time_limit=min(8, _time_left() - 8))
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
        
        # Stage 5: Block Coordinate Descent
        if _time_left() > 8:
            pts = _block_coordinate_descent(pts, time_limit=min(5, _time_left() - 5), max_rounds=2)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # If we still have time, do a final global polish on the best found
    if best_points is not None and _time_left() > 15:
        pts = _nelder_mead_optimize(best_points, time_limit=min(10, _time_left() - 10), maxiter=1000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
        pts = _block_coordinate_descent(best_points, time_limit=min(5, _time_left() - 5), max_rounds=2)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()

    if best_points is None:
        best_points = _fibonacci_points()
        
    return _normalize(best_points)
# EVOLVE-BLOCK-END
