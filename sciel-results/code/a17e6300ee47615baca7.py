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


def _facet_angles_and_indices(points):
    """
    Compute facet circumradius angles and return them along with facet indices.
    Returns: angles (array), facet_indices (array of simplex indices)
    """
    try:
        hull = ConvexHull(points)
    except:
        return np.array([np.pi]), np.array([0])
    
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
        return np.array([np.pi]), np.array([0])
    
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    h = np.sum(n * p1m, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    return angles, simplices[mask]


def _worst_facet_projection(points, time_limit=15, max_iter=200, step_scale=0.02):
    """
    Geometric worst-facet projection method:
    Iteratively identifies the convex hull facet with the largest spherical circumradius
    and projects the three defining vertices along their negative spherical gradients
    to shrink that specific facet.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    step_size = step_scale
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        angles, facets = _facet_angles_and_indices(points)
        if len(angles) == 0:
            break
        
        worst_idx = np.argmax(angles)
        worst_angle = angles[worst_idx]
        
        if worst_angle < 0.30:  # ~17.2 degrees, close to optimal
            break
        
        # Get the three vertices of the worst facet
        simplex = facets[worst_idx]
        i, j, k = simplex[0], simplex[1], simplex[2]
        
        p_i = points[i]
        p_j = points[j]
        p_k = points[k]
        
        # Compute facet normal
        v1 = p_j - p_i
        v2 = p_k - p_i
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # Ensure normal points outward
        if np.dot(n, p_i) < 0:
            n = -n
        
        # The facet center (on the sphere) is in the direction of n
        # To shrink the facet, we want to move vertices toward the facet center
        # but also consider the gradient of the circumradius with respect to each vertex
        
        # For a spherical triangle, the circumradius is minimized when the triangle
        # is more "equilateral" on the sphere. The gradient for each vertex points
        # away from the circumcenter of the spherical triangle.
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the point equidistant from all three vertices
        # It lies along the normal direction, at distance R from the origin
        # The circumcenter on the unit sphere is n itself (since the facet plane
        # is at distance h from origin, and the circumcenter is along n)
        
        # Actually, for a spherical triangle inscribed in a circle of angular radius r,
        # the circumcenter is at angular distance r from each vertex.
        # The direction from the origin to the circumcenter is n (the facet normal).
        # So the circumcenter on the unit sphere is n.
        
        circumcenter = n
        
        # Move each vertex slightly toward the circumcenter to reduce the circumradius
        # Use spherical linear interpolation (slerp)
        
        def slerp(p, q, t):
            """Spherical linear interpolation between p and q by fraction t."""
            dot_pq = np.dot(p, q)
            dot_pq = np.clip(dot_pq, -1.0, 1.0)
            theta = np.arccos(dot_pq)
            if theta < 1e-10:
                return p
            sin_theta = np.sin(theta)
            if sin_theta < 1e-10:
                return p
            w1 = np.sin((1 - t) * theta) / sin_theta
            w2 = np.sin(t * theta) / sin_theta
            return w1 * p + w2 * q
        
        # Adaptively scale the step based on how bad the facet is
        # Larger deviation from optimal -> larger step
        optimal_angle = 0.3193  # ~18.3 degrees in radians
        excess = max(0, worst_angle - optimal_angle)
        adaptive_step = step_size * (1.0 + 2.0 * excess)
        adaptive_step = min(adaptive_step, 0.15)
        
        new_p_i = slerp(p_i, circumcenter, adaptive_step)
        new_p_j = slerp(p_j, circumcenter, adaptive_step)
        new_p_k = slerp(p_k, circumcenter, adaptive_step)
        
        # Also consider moving vertices in the direction that reduces the
        # angular spread. For each vertex, the gradient of the circumradius
        # points in the direction away from the circumcenter projected onto
        # the tangent plane at that vertex.
        
        # Tangent gradient for vertex i:
        # grad_i = (circumcenter - p_i * dot(circumcenter, p_i)) / norm
        # This is the component of (circumcenter - p_i) tangent to the sphere at p_i
        
        def tangent_direction(p, target):
            """Project target direction onto tangent plane at p."""
            dot_pt = np.dot(p, target)
            tangent = target - dot_pt * p
            norm_t = np.linalg.norm(tangent)
            if norm_t < 1e-10:
                return np.zeros(3)
            return tangent / norm_t
        
        # The gradient of the circumradius w.r.t. vertex i points AWAY from the
        # circumcenter (to increase the circumradius), so we move in the negative
        # gradient direction, which is TOWARD the circumcenter.
        
        # But we also need to consider that moving one vertex affects the
        # circumradius in a non-trivial way. A simpler approach: move all three
        # vertices toward the circumcenter, which should reduce the circumradius.
        
        # Alternative: use the exact gradient. For a spherical triangle with
        # circumcenter C and vertices A, B, C_v, the circumradius r satisfies:
        # cos(r) = dot(A, C) = dot(B, C) = dot(C_v, C)
        # dr/dA = -sin(r) * (C - A*dot(A,C)) / sin(angle_AC) ... this gets complex.
        
        # Let's use a simpler heuristic: move vertices toward circumcenter,
        # but also consider the "pressure" from adjacent facets.
        
        points[i] = new_p_i
        points[j] = new_p_j
        points[k] = new_p_k
        points = _normalize(points)
        
        new_cost = _covering_radius(points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
            # Increase step slightly if making progress
            step_size = min(0.15, step_size * 1.05)
        else:
            # Revert
            points[i] = p_i
            points[j] = p_j
            points[k] = p_k
            # Decrease step
            step_size = max(0.001, step_size * 0.7)
            if step_size < 0.005:
                break
    
    return best_points


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


def _facet_smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate based on convex hull facet circumradii (vectorized).
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
    
    simplices = hull.simplices
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    mask = norm_n[:, 0] > 1e-10
    if not np.any(mask):
        return np.pi
    
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    h = np.sum(n * p1m, axis=1)
    h = np.clip(h, -1.0, 1.0)
    facet_angles = np.arccos(h)
    
    if len(facet_angles) == 0:
        return np.pi
    
    # Soft-max over facet angles
    beta_facet = beta * facet_angles
    max_val = np.max(beta_facet)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_facet - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _smooth_gradient_optimize(points, time_limit=30, beta=200.0, use_facet=False, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    if use_facet:
        grid = None
    elif grid is None:
        grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        if use_facet:
            return _facet_smooth_surrogate(params, beta=beta)
        else:
            return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _generate_test_grid(n_theta=40, n_phi=80):
    """Generate a grid of test points on the sphere."""
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _golden_spiral_points(offset=0.0):
    """Generate golden spiral points with offset for diversity."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _perturbed_fibonacci(seed=0):
    """Generate Fibonacci points with small random perturbations."""
    pts = _fibonacci_points()
    rng = np.random.RandomState(seed)
    noise = rng.randn(N, 3) * 0.05
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


def _spherical_lloyd_iteration(points, max_iter=100, time_limit=30, damping=0.3):
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
        angles = np.arccos(h)
        
        # Map back to original simplex indices
        valid_indices = np.where(mask)[0]
        
        # For each point, accumulate weighted normals from adjacent facets
        point_normal_sum = np.zeros((N, 3))
        point_weight_sum = np.zeros(N)
        
        for fi, si in enumerate(valid_indices):
            simplex = simplices[si]
            angle = angles[fi]
            for pi in simplex:
                point_normal_sum[pi] += angle * n_valid[fi]
                point_weight_sum[pi] += angle
        
        new_points = points.copy()
        for pi in range(N):
            if point_weight_sum[pi] > 1e-10:
                target = point_normal_sum[pi] / np.linalg.norm(point_normal_sum[pi])
                new_points[pi] = points[pi] + current_damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
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
                      options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 5000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _high_precision_polish(points, time_limit=10):
    """
    High-precision final polish: coordinate-wise Nelder-Mead with tight tolerances.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    # Coordinate-wise refinement: optimize each point one at a time
    for idx in range(N):
        if time.time() - start_time > time_limit:
            break
        
        other_points = np.delete(points, idx, axis=0)
        
        def obj_single(params):
            if time.time() - start_time > time_limit:
                return 1e6
            pt = params / np.linalg.norm(params)
            all_pts = np.vstack([other_points, pt.reshape(1, 3)])
            return _covering_radius_radians(all_pts)
        
        x0 = points[idx]
        res = minimize(obj_single, x0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-14, 'maxfev': 1000})
        new_pt = res.x / np.linalg.norm(res.x)
        points[idx] = new_pt
        
        new_cost = _covering_radius(points)
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = points.copy()
    
    # One more full Nelder-Mead pass with very tight tolerance
    if time.time() - start_time < time_limit - 2:
        def obj_full(params):
            if time.time() - start_time > time_limit:
                return 1e6
            pts = params.reshape(N, 3)
            pts = _normalize(pts)
            return _covering_radius_radians(pts)
        
        x0 = best_points.flatten()
        res = minimize(obj_full, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-13, 'maxfev': 2000})
        final_pts = _normalize(res.x.reshape(N, 3))
        final_cost = _covering_radius(final_pts)
        if final_cost < best_cost:
            best_cost = final_cost
            best_points = final_pts
    
    return best_points


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


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Precompute grid for smooth surrogate
    grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Golden spiral with offsets
    initializations.append(_golden_spiral_points(offset=0.25))
    initializations.append(_golden_spiral_points(offset=0.75))
    
    # Perturbed Fibonacci
    initializations.append(_perturbed_fibonacci(seed=1))
    initializations.append(_perturbed_fibonacci(seed=2))
    
    # Random with different seeds
    for seed in [42, 123, 777]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    for init_pts in initializations:
        if _time_left() < 30:
            break
        
        # Lloyd's relaxation
        pts = _spherical_lloyd_iteration(init_pts, max_iter=80, time_limit=min(12, _time_left() - 25), damping=0.3)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Worst-facet projection (new geometric method)
    if _time_left() > 25:
        pts = _worst_facet_projection(best_points, time_limit=min(15, _time_left() - 20), max_iter=200, step_scale=0.02)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Smooth gradient optimization (grid-based surrogate)
    if _time_left() > 25:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(12, _time_left() - 20), beta=200.0, use_facet=False, grid=grid)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Facet-based smooth gradient optimization
    if _time_left() > 20:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(10, _time_left() - 15), beta=100.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Nelder-Mead on true objective
    if _time_left() > 18:
        pts = _nelder_mead_optimize(best_points, time_limit=min(10, _time_left() - 12))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Simulated Annealing
    if _time_left() > 30:
        pts = _sa_optimize_covering(best_points, T_start=0.3, T_end=1e-4, n_iter=5000, time_limit=min(15, _time_left() - 15))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final Lloyd's polish
    if _time_left() > 15:
        pts = _spherical_lloyd_iteration(best_points, max_iter=50, time_limit=min(7, _time_left() - 10), damping=0.2)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Another round of worst-facet projection after other optimizations
    if _time_left() > 15:
        pts = _worst_facet_projection(best_points, time_limit=min(10, _time_left() - 10), max_iter=150, step_scale=0.01)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # High-precision polish
    if _time_left() > 12:
        pts = _high_precision_polish(best_points, time_limit=min(8, _time_left() - 5))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
