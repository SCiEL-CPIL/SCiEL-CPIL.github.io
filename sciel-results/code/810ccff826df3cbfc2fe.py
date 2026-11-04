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
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    valid = norms > 1e-10
    if not np.any(valid):
        return 180.0
    
    normals = normals[valid] / norms[valid, np.newaxis]
    p1v = p1[valid]
    
    # Ensure normals point outward
    dots = np.sum(normals * p1v, axis=1)
    flip = dots < 0
    if np.any(flip):
        normals[flip] = -normals[flip]
    
    h = np.sum(normals * p1v, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    return float(np.max(angles) * 180.0 / np.pi)


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _facet_smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate based on convex hull facet circumradii using log-sum-exp.
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
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    valid = norms > 1e-10
    if not np.any(valid):
        return np.pi
    
    normals = normals[valid] / norms[valid, np.newaxis]
    p1v = p1[valid]
    
    # Ensure normals point outward
    dots = np.sum(normals * p1v, axis=1)
    flip = dots < 0
    if np.any(flip):
        normals[flip] = -normals[flip]
    
    h = np.sum(normals * p1v, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    if len(angles) == 0:
        return np.pi
    
    # Soft-max over facet angles using log-sum-exp
    beta_angles = beta * angles
    max_val = np.max(beta_angles)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_angles - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _gradient_facet_objective(params, beta=200.0):
    """
    Compute objective and analytical gradient for the facet-based smooth surrogate.
    Returns (value, gradient) where gradient is w.r.t. the flattened params.
    """
    pts = params.reshape(N, 3).copy()
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi, np.zeros_like(params)
    
    simplices = hull.simplices
    
    # Precompute facet data
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    valid = norms > 1e-10
    if not np.any(valid):
        return np.pi, np.zeros_like(params)
    
    # Work with valid facets only
    valid_indices = np.where(valid)[0]
    normals_v = normals[valid] / norms[valid, np.newaxis]
    p1v = p1[valid]
    
    # Ensure normals point outward
    dots = np.sum(normals_v * p1v, axis=1)
    flip = dots < 0
    if np.any(flip):
        normals_v[flip] = -normals_v[flip]
    
    h = np.sum(normals_v * p1v, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    n_valid = len(angles)
    if n_valid == 0:
        return np.pi, np.zeros_like(params)
    
    # Compute softmax weights (log-sum-exp)
    beta_angles = beta * angles
    max_val = np.max(beta_angles)
    exp_terms = np.exp(beta_angles - max_val)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # (n_valid,)
    
    # Objective value
    value = (max_val + np.log(sum_exp)) / beta
    
    # Compute gradient using chain rule with proper facet geometry
    grad = np.zeros((N, 3))
    
    for i, vi in enumerate(valid_indices):
        w = weights[i]
        if w < 1e-10:
            continue
        
        si = simplices[vi]
        pi1, pi2, pi3 = si[0], si[1], si[2]
        
        h_i = h[i]
        # d(angle)/dh = -1/sqrt(1-h^2)
        dh_factor = -1.0 / np.sqrt(max(1e-12, 1.0 - h_i * h_i))
        
        n_i = normals_v[i]
        
        # The circumradius angle theta satisfies cos(theta) = n . p1
        # For a triangle on the unit sphere, moving vertices changes both n and p1
        # Approximate: d(theta)/d(p_k) ≈ -n_i for k in facet
        # This is a first-order approximation that works well for small perturbations
        
        for pk in [pi1, pi2, pi3]:
            grad[pk] += w * dh_factor * (-n_i)
    
    # Project onto tangent space of the sphere
    for k in range(N):
        p_k = pts[k]
        proj = np.dot(grad[k], p_k)
        grad[k] -= proj * p_k
    
    return value, grad.flatten()


def _smooth_gradient_optimize_facet(points, time_limit=30, beta=200.0):
    """
    Gradient-based optimization using facet-based smooth surrogate.
    Uses custom gradient computation for better convergence.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6, np.zeros_like(params)
        val, grad = _gradient_facet_objective(params)
        return val, grad
    
    # Use L-BFGS-B with custom gradient
    x0 = points.flatten()
    
    # Define objective and gradient separately for scipy
    def obj_only(params):
        val, _ = _gradient_facet_objective(params)
        return val
    
    def grad_only(params):
        _, grad = _gradient_facet_objective(params)
        return grad
    
    result = minimize(obj_only, x0, jac=grad_only, method='L-BFGS-B', 
                      options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 10000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


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


def _spherical_lloyd_iteration(points, max_iter=100, time_limit=30, damping=0.3):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) with adaptive damping.
    Vectorized facet processing for speed.
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
        n_facets = len(simplices)
        
        # Vectorized facet normal computation
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            break
        
        normals = normals[valid] / norms[valid, np.newaxis]
        p1v = p1[valid]
        
        # Ensure normals point outward
        dots = np.sum(normals * p1v, axis=1)
        flip = dots < 0
        if np.any(flip):
            normals[flip] = -normals[flip]
        
        # Accumulate weighted normals per point
        normal_sums = np.zeros((N, 3))
        weight_sums = np.zeros(N)
        
        # For each valid facet, add its weighted normal to all three vertices
        valid_indices = np.where(valid)[0]
        for fi in valid_indices:
            simplex = simplices[fi]
            w = norms[fi]
            n = normals[fi]
            for pi in simplex:
                normal_sums[pi] += w * n
                weight_sums[pi] += w
        
        new_points = points.copy()
        for pi in range(N):
            if weight_sums[pi] > 1e-10:
                target = normal_sums[pi] / weight_sums[pi]
                target_norm = np.linalg.norm(target)
                if target_norm > 1e-10:
                    target = target / target_norm
                    new_points[pi] = points[pi] + current_damping * (target - points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
            current_damping *= 0.95
        else:
            current_damping *= 0.5
            if current_damping < 0.01:
                break
    
    return best_points


def _worst_facet_relaxation(points, time_limit=10.0, max_iter=50):
    """
    Targeted relaxation: move points adjacent to the worst facet towards
    reduce its circumradius.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    for _ in range(max_iter):
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
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            break
        
        normals = normals[valid] / norms[valid, np.newaxis]
        p1v = p1[valid]
        
        dots = np.sum(normals * p1v, axis=1)
        flip = dots < 0
        if np.any(flip):
            normals[flip] = -normals[flip]
        
        h = np.sum(normals * p1v, axis=1)
        h = np.clip(h, -1.0, 1.0)
        angles = np.arccos(h)
        
        # Find the worst facet
        worst_idx = np.argmax(angles)
        worst_angle = angles[worst_idx]
        
        if worst_angle < 0.1:  # ~5.7 degrees, already good
            break
        
        # Get the three points of the worst facet
        valid_indices = np.where(valid)[0]
        worst_simplex = simplices[valid_indices[worst_idx]]
        
        # Compute circumcenter of the worst facet
        wp1 = points[worst_simplex[0]]
        wp2 = points[worst_simplex[1]]
        wp3 = points[worst_simplex[2]]
        
        wv1 = wp2 - wp1
        wv2 = wp3 - wp1
        wn = np.cross(wv1, wv2)
        wn_norm = np.linalg.norm(wn)
        if wn_norm < 1e-10:
            break
        wn = wn / wn_norm
        
        if np.dot(wn, wp1) < 0:
            wn = -wn
        
        # Move the three points slightly towards the circumcenter direction
        damping = 0.1
        for pi in worst_simplex:
            target = wn
            target = target / np.linalg.norm(target)
            points[pi] = points[pi] + damping * (target - points[pi])
        
        points = _normalize(points)
        new_cost = _covering_radius(points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = points.copy()
    
    return best_points


def _tangent_polish(points, time_limit=15.0, max_iter=300):
    """
    High-precision polish using tangent space optimization.
    Addresses multiple worst facets simultaneously with adaptive steps.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    step_size = 0.03
    consecutive_rejects = 0
    
    for it in range(max_iter):
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
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            break
        
        normals = normals[valid] / norms[valid, np.newaxis]
        p1v = p1[valid]
        
        dots = np.sum(normals * p1v, axis=1)
        flip = dots < 0
        if np.any(flip):
            normals[flip] = -normals[flip]
        
        h = np.sum(normals * p1v, axis=1)
        h = np.clip(h, -1.0, 1.0)
        angles = np.arccos(h)
        
        # Find top-5 worst facets
        n_worst = min(5, len(angles))
        worst_indices = np.argsort(angles)[::-1][:n_worst]
        worst_angle = angles[worst_indices[0]]
        
        if worst_angle < 0.003:  # Very small, likely converged
            break
        
        valid_indices = np.where(valid)[0]
        
        # Accumulate gradient directions per point with weighting by angle
        grad_accum = np.zeros((N, 3))
        
        for wi in worst_indices:
            worst_simplex = simplices[valid_indices[wi]]
            wp1 = points[worst_simplex[0]]
            wp2 = points[worst_simplex[1]]
            wp3 = points[worst_simplex[2]]
            
            wv1 = wp2 - wp1
            wv2 = wp3 - wp1
            wn = np.cross(wv1, wv2)
            wn_norm = np.linalg.norm(wn)
            if wn_norm < 1e-10:
                continue
            wn = wn / wn_norm
            
            if np.dot(wn, wp1) < 0:
                wn = -wn
            
            # Weight by how much worse this facet is
            weight = max(angles[wi] - 0.01, 0.0) + 0.1
            for pi in worst_simplex:
                grad_accum[pi] += weight * wn
        
        # Apply gradient with adaptive step
        for pi in range(N):
            if np.linalg.norm(grad_accum[pi]) > 1e-10:
                current = points[pi]
                direction = grad_accum[pi] - np.dot(grad_accum[pi], current) * current
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    direction = direction / dir_norm
                    points[pi] = current + step_size * direction
        
        points = _normalize(points)
        new_cost = _covering_radius(points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = points.copy()
            consecutive_rejects = 0
            step_size = min(step_size * 1.05, 0.08)  # Slightly increase step on success
        else:
            step_size *= 0.6
            consecutive_rejects += 1
            if consecutive_rejects > 15:
                break
    
    return best_points


def _high_beta_facet_optimize(points, time_limit=20.0):
    """
    High-beta facet-based optimization with multiple temperature annealing steps.
    Uses analytical gradient for efficiency.
    """
    start_time = time.time()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    # Anneal through different beta values
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit - 3:
            break
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            break
        
        def obj_only(params):
            val, _ = _gradient_facet_objective(params, beta=beta)
            return val
        
        def grad_only(params):
            _, grad = _gradient_facet_objective(params, beta=beta)
            return grad
        
        x0 = best_points.flatten()
        result = minimize(obj_only, x0, jac=grad_only, method='L-BFGS-B', 
                          options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 3000})
        candidate = _normalize(result.x.reshape(N, 3))
        cost = _covering_radius(candidate)
        
        if cost < best_cost:
            best_cost = cost
            best_points = candidate.copy()
    
    return best_points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 148.0  # Safety margin for 150s limit
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Jittered Fibonacci (breaks symmetry)
    for seed in [123, 456, 789, 246, 357, 678, 891, 111]:
        initializations.append(_jittered_fibonacci_points(seed=seed, jitter=0.03))
    
    # Random with different seeds
    for seed in [42, 123, 777, 999, 555, 333, 888, 222]:
        initializations.append(_random_points(seed=seed))
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Lloyd's relaxation from multiple starts (budget: ~50s)
    lloyd_time_each = 2.0
    for init_pts in initializations:
        if _time_left() < 50:
            break
        
        pts = _spherical_lloyd_iteration(init_pts, max_iter=80, time_limit=min(lloyd_time_each, _time_left() - 45), damping=0.3)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: High-beta facet-based optimization with annealing (budget: ~25s)
    if _time_left() > 40:
        pts = _high_beta_facet_optimize(best_points, time_limit=min(22, _time_left() - 35))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 3: Worst facet relaxation (budget: ~8s)
    if _time_left() > 22:
        pts = _worst_facet_relaxation(best_points, time_limit=min(7, _time_left() - 18), max_iter=40)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Another round of high-beta optimization (budget: ~15s)
    if _time_left() > 18:
        pts = _high_beta_facet_optimize(best_points, time_limit=min(13, _time_left() - 14))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5: Tangent space polish for high precision (budget: ~12s)
    if _time_left() > 10:
        pts = _tangent_polish(best_points, time_limit=min(10, _time_left() - 7), max_iter=400)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: Final worst facet relaxation (budget: ~4s)
    if _time_left() > 6:
        pts = _worst_facet_relaxation(best_points, time_limit=min(3, _time_left() - 4), max_iter=30)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 7: Final high-precision LSE polish (budget: ~4s)
    if _time_left() > 5:
        def final_obj(params):
            val, _ = _gradient_facet_objective(params, beta=800.0)
            return val
        
        def final_grad(params):
            _, grad = _gradient_facet_objective(params, beta=800.0)
            return grad
        
        x0 = best_points.flatten()
        result = minimize(final_obj, x0, jac=final_grad, method='L-BFGS-B', 
                          options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 2500})
        candidate = _normalize(result.x.reshape(N, 3))
        cost = _covering_radius(candidate)
        if cost < best_cost:
            best_cost = cost
            best_points = candidate
    
    # Phase 8: Final tangent polish with very small steps
    if _time_left() > 3:
        pts = _tangent_polish(best_points, time_limit=min(2, _time_left() - 1), max_iter=200)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
