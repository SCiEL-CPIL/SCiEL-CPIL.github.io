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
    Smooth surrogate based on convex hull facet circumradii.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
    
    facet_angles = []
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
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
        facet_angles.append(angle)
    
    if not facet_angles:
        return np.pi
    
    facet_angles = np.array(facet_angles)
    
    # Soft-max over facet angles
    beta_facet = beta * facet_angles
    max_val = np.max(beta_facet)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_facet - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _smooth_gradient_optimize(points, time_limit=30, beta=100.0, use_facet=False, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    if use_facet:
        grid = None
    elif grid is None:
        grid = _generate_test_grid(n_theta=30, n_phi=60)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        if use_facet:
            return _facet_smooth_surrogate(params, beta=beta)
        else:
            return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000})
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
        
        new_points = points.copy()
        
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            normal_sum = np.zeros(3)
            weight_sum = 0.0
            for fi in facets:
                simplex = hull.simplices[fi]
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
                
                # Weight by the angle (larger angles get more weight)
                w = angle
                normal_sum += w * n
                weight_sum += w
            
            if weight_sum > 1e-10:
                target = normal_sum / np.linalg.norm(normal_sum)
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
            # Adaptive damping: increase if improving fast
            current_damping = min(0.8, current_damping * 1.1)
        else:
            # Decrease damping and try again
            current_damping *= 0.5
            if current_damping < 0.01:
                break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=20, maxiter=500):
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


def _gradient_descent_smooth(points, grid, time_limit=15, beta=200.0, lr=0.01):
    """
    Gradient descent on the smooth surrogate for fine-tuning.
    """
    start_time = time.time()
    pts = _normalize(points).copy()
    
    def _surrogate_and_grad(pts):
        pts_n = _normalize(pts)
        dots = np.clip(grid @ pts_n.T, -1.0, 1.0)
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
        exp_max = np.exp(beta_soft_min - max_val)
        sum_exp_max = np.sum(exp_max)
        soft_max = (max_val + np.log(sum_exp_max)) / beta
        
        # Gradient: d(soft_max)/d(pts)
        # soft_max = (1/beta) * logsumexp(beta * soft_min_angles)
        # d(soft_max)/d(soft_min_angles[i]) = exp(beta*soft_min_angles[i] - max_val) / sum_exp_max
        weights_max = exp_max / sum_exp_max  # (M,)
        
        # d(soft_min_angles[i])/d(pts) 
        # soft_min_angles[i] = -1/beta * logsumexp(-beta * angles[i,:])
        # d(soft_min_angles[i])/d(angles[i,j]) = exp(-beta*angles[i,j] - logsumexp_neg[i]) / 1  ... 
        # Actually: d(-logsumexp(-beta*angles))/d(angles[j]) = beta * exp(-beta*angles[j]) / sum(exp(-beta*angles))
        # = beta * exp_terms[i,j] / sum_exp[i]
        # So d(soft_min_angles[i])/d(angles[i,j]) = -exp_terms[i,j] / sum_exp[i] * (-1) ... let me redo
        # soft_min = -1/beta * LSE(-beta * angles)
        # d(soft_min)/d(angles[j]) = -1/beta * d(LSE)/d(angles[j]) = -1/beta * (-beta) * exp(-beta*angles[j])/sum = exp(-beta*angles[j])/sum
        # = exp_terms[i,j] / sum_exp[i]
        
        # d(angles[i,j])/d(pts[j]) where angles = arccos(grid @ pts[j])
        # d(arccos(x))/dx = -1/sqrt(1-x^2)
        # x = grid[i] @ pts[j]
        # d(x)/d(pts[j]) = grid[i]
        # So d(angles[i,j])/d(pts[j]) = -grid[i] / sqrt(1 - x^2)
        
        sqrt_terms = np.sqrt(np.clip(1.0 - dots**2, 1e-10, None))  # (M, N)
        
        # Combine: d(soft_max)/d(pts[j]) = sum_i weights_max[i] * (exp_terms[i,j]/sum_exp[i]) * (-grid[i] / sqrt_terms[i,j])
        # = -sum_i (weights_max[i] * exp_terms[i,j] / (sum_exp[i] * sqrt_terms[i,j])) * grid[i]
        
        coeff = weights_max[:, np.newaxis] * exp_terms / (sum_exp[:, np.newaxis] * sqrt_terms)  # (M, N)
        grad = -(coeff.T @ grid)  # (N, 3)
        
        # Project gradient to tangent space
        for j in range(N):
            grad[j] -= np.dot(grad[j], pts_n[j]) * pts_n[j]
        
        return soft_max, grad
    
    best_cost, _ = _surrogate_and_grad(pts)
    best_pts = pts.copy()
    
    for step in range(2000):
        if time.time() - start_time > time_limit:
            break
        
        cost, grad = _surrogate_and_grad(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_pts = pts.copy()
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
        
        step_size = lr / (1.0 + step * 0.01)
        pts_new = pts - step_size * grad
        pts_new = _normalize(pts_new)
        
        new_cost, _ = _surrogate_and_grad(pts_new)
        if new_cost < cost:
            pts = pts_new
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return best_pts


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
                      options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-9})
    new_pt = result.x / np.linalg.norm(result.x)
    points[idx] = new_pt
    return points


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
        pts = _spherical_lloyd_iteration(init_pts, max_iter=50, time_limit=min(15, _time_left() - 25), damping=0.3)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Multi-pass refinement: alternate between smooth gradient and Nelder-Mead
    for pass_num in range(3):
        if _time_left() < 20:
            break
        
        # Smooth gradient optimization (grid-based surrogate)
        if _time_left() > 25:
            pts = _smooth_gradient_optimize(best_points, time_limit=min(18, _time_left() - 20), beta=200.0, use_facet=False, grid=grid)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
        
        # Gradient descent on smooth surrogate for fine-tuning
        if _time_left() > 20:
            pts = _gradient_descent_smooth(best_points, grid, time_limit=min(10, _time_left() - 15), beta=200.0, lr=0.005)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
        
        # Nelder-Mead on true objective
        if _time_left() > 15:
            pts = _nelder_mead_optimize(best_points, time_limit=min(12, _time_left() - 12), maxiter=400)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Final Lloyd's polish
    if _time_left() > 10:
        pts = _spherical_lloyd_iteration(best_points, max_iter=30, time_limit=min(5, _time_left() - 8), damping=0.2)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final smooth gradient polish
    if _time_left() > 12:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 10), beta=300.0, use_facet=False, grid=grid)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final Nelder-Mead polish with tighter tolerance
    if _time_left() > 8:
        pts = _nelder_mead_optimize(best_points, time_limit=min(5, _time_left() - 5), maxiter=600)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    # Verify no coincident points and valid configuration
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_points()
    
    return best_points
# EVOLVE-BLOCK-END
