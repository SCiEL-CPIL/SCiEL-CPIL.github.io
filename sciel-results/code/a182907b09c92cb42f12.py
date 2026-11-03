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
    Smooth differentiable surrogate based on convex hull facet circumradii.
    Uses soft-max over facet angles to approximate the maximum.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    angles = _facet_circumradius_angles(pts)
    
    if len(angles) == 0:
        return np.pi/2
    
    # Soft-max: logsumexp(beta * angles) / beta
    beta_angles = beta * angles
    max_val = np.max(beta_angles)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_angles - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _smooth_gradient_optimize(points, time_limit=30, beta=50.0):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    Uses facet-based smooth surrogate aligned with actual scoring.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _facet_smooth_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000})
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


def _riesz_s_energy(points, s=2.0):
    """Compute Riesz s-energy: sum of 1/|p_i - p_j|^s over all pairs."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    return np.sum(1.0 / (dists ** s))


def _riesz_s_gradient(points, s=2.0):
    """Compute gradient of Riesz s-energy w.r.t. each point."""
    grad = np.zeros_like(points)
    for i in range(N):
        diff = points[i] - points  # shape (N, 3)
        dists = np.linalg.norm(diff, axis=1)
        dists[i] = 1.0
        # d/dp_i [sum_j 1/|p_i-p_j|^s] = -s * sum_j (p_i - p_j) / |p_i-p_j|^(s+2)
        weights = s / (dists ** (s + 2))
        weights[i] = 0.0
        grad[i] = -np.sum(weights[:, np.newaxis] * diff, axis=0)
    return grad


def _riesz_refine(points, time_limit=15, s=2.0, lr=0.01):
    """Gradient descent on Riesz s-energy to spread points."""
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_energy = _riesz_s_energy(points, s)
    
    for step in range(500):
        if time.time() - start_time > time_limit:
            break
        
        grad = _riesz_s_gradient(points, s)
        # Project gradient to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], points[i]) * points[i]
        
        points = points - lr * grad
        points = _normalize(points)
        
        energy = _riesz_s_energy(points, s)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return _normalize(best_points)


def _multi_beta_lse_refine(points, time_limit=30):
    """
    Multi-stage beta cascade polish using L-BFGS-B on the smooth log-sum-exp surrogate.
    Starts with low beta (broad basin) and increases to high beta (sharp optimum).
    """
    start_time = time.time()
    pts = _normalize(points).copy()
    
    betas = [10.0, 30.0, 100.0, 300.0, 500.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            break
        
        def objective(params, b=beta):
            if time.time() - start_time > time_limit:
                return 1e6
            p = params.reshape(N, 3)
            p = _normalize(p)
            angles = _facet_circumradius_angles(p)
            if len(angles) == 0:
                return np.pi / 2
            ba = b * angles
            max_val = np.max(ba)
            logsumexp = max_val + np.log(np.sum(np.exp(ba - max_val)))
            return logsumexp / b
        
        x0 = pts.flatten()
        result = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1500})
        pts = _normalize(result.x.reshape(N, 3))
    
    return pts


def _facet_weighted_descent(points, time_limit=20, lr=0.02, beta=30.0):
    """
    Gradient descent that only moves points on high-circumradius facets.
    Uses softmax weighting to focus on worst facets.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    for step in range(300):
        if time.time() - start_time > time_limit:
            break
        
        angles = _facet_circumradius_angles(points)
        if len(angles) == 0:
            break
        
        # Softmax weights for facets
        beta_angles = beta * angles
        max_val = np.max(beta_angles)
        weights = np.exp(beta_angles - max_val)
        weights /= np.sum(weights)
        
        # For each point, compute weighted sum of facet normals
        point_grad = np.zeros((N, 3))
        try:
            hull = ConvexHull(points)
        except:
            break
        
        for fi, simplex in enumerate(hull.simplices):
            if weights[fi] < 1e-4:
                continue
            
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
            
            # Move points toward facet normal direction
            for pi in simplex:
                point_grad[pi] += weights[fi] * n
        
        # Project to tangent space and apply
        for i in range(N):
            point_grad[i] -= np.dot(point_grad[i], points[i]) * points[i]
        
        points = points - lr * point_grad
        points = _normalize(points)
        
        cost = _covering_radius(points)
        if cost < best_cost:
            best_cost = cost
            best_points = points.copy()
        else:
            lr *= 0.95
            if lr < 1e-5:
                break
    
    return _normalize(best_points)


def _simulated_annealing(points, time_limit=25, T_start=0.1, T_end=0.01, alpha=0.98):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    T = T_start
    rng = np.random.RandomState(42)
    
    for step in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        candidate = points.copy()
        idx = rng.randint(N)
        noise = rng.randn(3) * T
        candidate[idx] = points[idx] + noise
        candidate[idx] = candidate[idx] / np.linalg.norm(candidate[idx])
        
        cost = _covering_radius(candidate)
        delta = cost - best_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            points = candidate
            if cost < best_cost:
                best_cost = cost
                best_points = points.copy()
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    return _normalize(best_points)


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
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
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
    
    # Phase 1: High-exponent Riesz energy refinement (s=20) for packing-based initialization
    # High s pushes points to maximize minimum pairwise separation (packing),
    # which indirectly minimizes covering radius via packing-covering duality.
    for init_pts in initializations:
        if _time_left() < 45:
            break
        
        # Use high exponent s=20 for strong repulsion / packing behavior
        pts = _riesz_refine(init_pts, time_limit=min(6, _time_left() - 40), s=20.0, lr=0.005)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Multi-stage beta cascade polish (10 -> 500)
    # Low beta first finds the right basin, high beta sharpens to high precision.
    if _time_left() > 35:
        pts = _multi_beta_lse_refine(best_points, time_limit=min(25, _time_left() - 15))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 3: Facet-weighted descent for additional refinement
    if _time_left() > 15:
        pts = _facet_weighted_descent(best_points, time_limit=min(10, _time_left() - 10), lr=0.01, beta=50.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Final Nelder-Mead polish if time permits
    if _time_left() > 5:
        pts = _nelder_mead_optimize(best_points, time_limit=min(4, _time_left() - 2), maxiter=1000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
