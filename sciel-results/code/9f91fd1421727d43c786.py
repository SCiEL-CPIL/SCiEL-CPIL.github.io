"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize_points(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius_radians(points):
    """Compute the true covering radius in radians using hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi
        
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return np.pi
        
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    cos_r = np.sum(n * p1m, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    return float(np.max(radii))


def _generate_test_grid(n_theta=60, n_phi=120):
    """
    Generate a dense grid of test points on the sphere.
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


# Pre-compute grid once
_GRID = _generate_test_grid(n_theta=60, n_phi=120)


def _grid_surrogate(params, beta=100.0, grid=None):
    """
    Smooth differentiable surrogate for the covering radius using a grid.
    soft-max over grid of soft-min over points of angular distance.
    """
    if grid is None:
        grid = _GRID
    
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    # Angular distances: grid (M,3) x pts (N,3) -> (M,N)
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


def _grid_surrogate_grad(params, beta=100.0, grid=None):
    """
    Compute value and gradient of the grid surrogate using finite differences.
    """
    val = _grid_surrogate(params, beta=beta, grid=grid)
    
    grad = np.zeros_like(params)
    h = 1e-5
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + h
        val_plus = _grid_surrogate(params, beta=beta, grid=grid)
        params[i] = orig - h
        val_minus = _grid_surrogate(params, beta=beta, grid=grid)
        params[i] = orig
        grad[i] = (val_plus - val_minus) / (2 * h)
    
    return val, grad


def _facet_shrink_polish(points, time_limit=15.0):
    """
    Geometric relaxation: move worst facet vertices toward circumcenter.
    """
    start_time = time.time()
    step = 0.03
    
    for it in range(5000):
        if time.time() > start_time + time_limit:
            break
            
        current_radius, worst_facet, worst_center = _get_worst_facet(points)
        
        if worst_facet is None:
            break
            
        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]
        
        def move_vertex(v, c, step):
            grad = c - np.dot(c, v) * v
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                return v
            grad = grad / grad_norm
            v_new = v + step * grad
            return v_new / np.linalg.norm(v_new)
        
        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)
        
        if it % 50 == 0:
            step *= 0.97
            if step < 0.0005:
                step = 0.0005

    return points


def _get_worst_facet(points):
    """Find the worst facet and its circumcenter."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, None, None
        
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi, None, None
        
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return np.pi, None, None
        
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    cos_r = np.sum(n * p1m, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    max_idx = np.argmax(radii)
    max_r = radii[max_idx]
    
    # Get the simplex index
    valid_indices = np.where(mask)[0]
    worst_simplex_idx = valid_indices[max_idx]
    worst_facet = simplices[worst_simplex_idx]
    worst_center = n[max_idx]
    
    return max_r, worst_facet, worst_center


def _repulsive_energy(params, s=20.0):
    """
    Riesz s-energy with high exponent to maximize minimum pairwise separation.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    dots = np.dot(pts, pts.T)
    dots = np.clip(dots, -1.0, 1.0)
    
    dist_sq = 2.0 - 2.0 * dots
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    dist_sq = np.maximum(dist_sq, 1e-10)
    
    return np.sum(dist_sq ** (-s / 2.0))


def _repulsive_energy_grad(params, s=20.0):
    """
    Compute value and gradient of Riesz s-energy via finite differences.
    """
    val = _repulsive_energy(params, s=s)
    
    grad = np.zeros_like(params)
    h = 1e-5
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + h
        e_plus = _repulsive_energy(params, s=s)
        params[i] = orig - h
        e_minus = _repulsive_energy(params, s=s)
        params[i] = orig
        grad[i] = (e_plus - e_minus) / (2 * h)
    
    return val, grad


def _gradient_descent_phase(points, time_limit=20.0, beta=100.0, use_repulsive=False, s=20.0, use_grad=True):
    """
    Perform gradient descent using L-BFGS-B on the grid surrogate or Riesz energy.
    """
    start_time = time.time()
    
    if use_repulsive:
        if use_grad:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _repulsive_energy(params, s=s)
            def gradient(params):
                if time.time() - start_time > time_limit:
                    return np.zeros_like(params)
                _, g = _repulsive_energy_grad(params, s=s)
                return g
        else:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _repulsive_energy(params, s=s)
            gradient = None
    else:
        if use_grad:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _grid_surrogate(params, beta=beta)
            def gradient(params):
                if time.time() - start_time > time_limit:
                    return np.zeros_like(params)
                _, g = _grid_surrogate_grad(params, beta=beta)
                return g
        else:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _grid_surrogate(params, beta=beta)
            gradient = None
    
    x0 = points.flatten()
    
    try:
        if gradient is not None:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200})
        else:
            result = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200})
        best_points = _normalize_points(result.x.reshape(N, 3))
        return best_points
    except Exception:
        return points


def _multi_restart_optimize(points, deadline, num_restarts=3):
    """
    Try multiple perturbations and keep the best result.
    """
    best_points = points.copy()
    best_radius = _covering_radius_radians(points)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        perturbation_scale = 0.05 * (restart + 1)
        noise = np.random.randn(N, 3) * perturbation_scale
        perturbed = _normalize_points(points + noise)
        
        optimized = _gradient_descent_phase(perturbed, time_limit=12.0, beta=50.0, use_grad=False)
        
        polished = _facet_shrink_polish(optimized, time_limit=6.0)
        
        radius = _covering_radius_radians(polished)
        if radius < best_radius:
            best_radius = radius
            best_points = polished.copy()
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for safety
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius_radians(points)
    
    # Phase 1: Coarse optimization with low beta
    if time.time() < deadline - 80.0:
        points = _gradient_descent_phase(points, time_limit=25.0, beta=20.0, use_grad=False)
        current_radius = _covering_radius_radians(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 2: Medium beta
    if time.time() < deadline - 50.0:
        points = _gradient_descent_phase(points, time_limit=20.0, beta=80.0, use_grad=False)
        current_radius = _covering_radius_radians(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 3: High beta polishing
    if time.time() < deadline - 25.0:
        points = _gradient_descent_phase(points, time_limit=15.0, beta=200.0, use_grad=False)
        current_radius = _covering_radius_radians(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 4: Repulsive energy refinement
    if time.time() < deadline - 15.0:
        repulsive_points = best_points.copy()
        repulsive_points = _gradient_descent_phase(repulsive_points, time_limit=10.0, use_repulsive=True, s=30.0, use_grad=False)
        repulsive_radius = _covering_radius_radians(repulsive_points)
        if repulsive_radius < best_radius:
            best_radius = repulsive_radius
            best_points = repulsive_points.copy()
            if time.time() < deadline - 5.0:
                polished = _facet_shrink_polish(best_points, time_limit=5.0)
                pol_radius = _covering_radius_radians(polished)
                if pol_radius < best_radius:
                    best_radius = pol_radius
                    best_points = polished.copy()
    
    # Phase 5: Multi-restart exploration
    if time.time() < deadline - 20.0:
        best_points, best_radius = _multi_restart_optimize(best_points, deadline, num_restarts=2)
    
    # Phase 6: Final facet shrink polish
    if time.time() < deadline - 3.0:
        polished = _facet_shrink_polish(best_points, time_limit=10.0)
        pol_radius = _covering_radius_radians(polished)
        if pol_radius < best_radius:
            best_radius = pol_radius
            best_points = polished.copy()
    
    # Final high-beta gradient polish if time allows
    if time.time() < deadline - 2.0:
        final_points = _gradient_descent_phase(best_points, time_limit=3.0, beta=400.0, use_grad=False)
        final_radius = _covering_radius_radians(final_points)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_points.copy()

    # Final check
    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return best_points
    else:
        return points
# EVOLVE-BLOCK-END
