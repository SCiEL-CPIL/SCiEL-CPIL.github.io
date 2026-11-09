# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _points_to_sphere(params):
    """Convert spherical coordinates (theta, phi per point) to 3D unit sphere points."""
    n = params.shape[0]
    theta = params[:, 0]  # polar angle [0, pi]
    phi = params[:, 1]    # azimuthal angle [0, 2*pi]
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.column_stack([x, y, z])

def _points_to_sphere_params(points):
    """Convert 3D points to spherical coordinates (theta, phi)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    theta = np.arccos(np.clip(z / np.sqrt(x**2 + y**2 + z**2 + 1e-300), -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = (phi + 2 * np.pi) % (2 * np.pi)  # Ensure phi in [0, 2*pi)
    return np.column_stack([theta, phi])

def _smooth_objective_sphere(params, n=14, beta=20.0, alpha=20.0):
    """
    Smooth surrogate on spherical coordinates.
    Minimizes - (log(softmin(d)) - log(softmax(d)))
    """
    points = _points_to_sphere(params)
    dists = pdist(points)
    
    if np.any(dists < 1e-12):
        return 1e6
    
    neg_beta_d = -beta * dists
    max_neg_beta_d = np.max(neg_beta_d)
    log_sum_exp_neg = max_neg_beta_d + np.log(np.sum(np.exp(neg_beta_d - max_neg_beta_d)))
    log_softmin = -1.0 / beta * log_sum_exp_neg
    
    alpha_d = alpha * dists
    max_alpha_d = np.max(alpha_d)
    log_sum_exp_pos = max_alpha_d + np.log(np.sum(np.exp(alpha_d - max_alpha_d)))
    log_softmax = 1.0 / alpha * log_sum_exp_pos
    
    objective_value = -(log_softmin - log_softmax)
    
    if not np.isfinite(objective_value):
        return 1e6
        
    return objective_value

def _true_objective_sphere(params, n=14):
    """True objective on spherical coordinates (to minimize)."""
    points = _points_to_sphere(params)
    return -_score(points)

def _make_icosahedron_plus_2():
    """Icosahedron (12 verts) + 2 points at poles of a face."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            verts.append([0, s1, s2 * phi])
            verts.append([s1, s2 * phi, 0])
            verts.append([s2 * phi, 0, s1])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_cuboctahedron_plus_2():
    """Cuboctahedron (12 verts) + 2 points."""
    verts = []
    for x in [1, -1]:
        for y in [1, -1]:
            verts.append([x, y, 0])
            verts.append([x, 0, y])
            verts.append([0, x, y])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([-1.0, 0.0, 0.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_random_unit(n=14, d=3, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    points = rng.randn(n, d)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    return points

def _polish_sphere(params, deadline, max_iter=3000):
    """Nelder-Mead polish on spherical coordinates."""
    n = params.shape[0]
    try:
        res = minimize(_true_objective_sphere, params, method='Nelder-Mead',
                       args=(n,),
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if not np.any(np.isnan(res.x)):
            return res.x
    except Exception:
        pass
    return params

def _polish_cartesian(points, deadline):
    """Nelder-Mead polish in Cartesian coordinates as final refinement."""
    n, d = points.shape
    try:
        res = minimize(lambda x: -_score(x.reshape(n, d)), points.flatten(), method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if not np.any(np.isnan(res.x)):
            return res.x.reshape(n, d)
    except Exception:
        pass
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical coordinates for constrained optimization on the unit sphere.
    """
    n = 14
    start_time = time.time()
    deadline = start_time + 580  # 630s limit, keep 50s margin
    
    best_score = -1.0
    best_points = None
    
    def _update_best(points):
        nonlocal best_score, best_points
        if points is not None and np.all(np.isfinite(points)):
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
    
    # Generate diverse initial guesses in Cartesian, convert to sphere params
    guesses_cart = []
    guesses_cart.append(_make_icosahedron_plus_2())
    guesses_cart.append(_make_cuboctahedron_plus_2())
    for i in range(8):
        guesses_cart.append(_make_random_unit(n, 3, seed=1000 + i * 137))
    
    # Evaluate initial scores
    for gc in guesses_cart:
        _update_best(gc)
    
    # Phase 1: Differential Evolution on spherical coordinates (global search)
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0, np.pi), (0, 2 * np.pi)] * n
    
    if time.time() < deadline - 120:
        try:
            de_result = differential_evolution(
                _true_objective_sphere, bounds,
                args=(n,),
                maxiter=80,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol',
                updating='immediate'
            )
            if not np.any(np.isnan(de_result.x)):
                de_points = _points_to_sphere(de_result.x)
                _update_best(de_points)
        except Exception:
            pass
    
    # Phase 2: Multi-start Nelder-Mead on spherical coordinates
    if time.time() < deadline - 60:
        for i, gc in enumerate(guesses_cart):
            if time.time() > deadline - 30:
                break
            
            params = _points_to_sphere_params(gc)
            
            # Smooth optimization with L-BFGS-B on sphere params
            try:
                res_smooth = minimize(_smooth_objective_sphere, params, method='L-BFGS-B',
                                      args=(n,),
                                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                if not np.any(np.isnan(res_smooth.x)):
                    pts_smooth = _points_to_sphere(res_smooth.x)
                    _update_best(pts_smooth)
            except Exception:
                pass
            
            # Nelder-Mead polish on sphere
            if time.time() > deadline - 20:
                break
            try:
                params_start = res_smooth.x if not np.any(np.isnan(res_smooth.x)) else params
                params_polished = _polish_sphere(params_start, deadline)
                pts_polished = _points_to_sphere(params_polished)
                _update_best(pts_polished)
            except Exception:
                pass
    
    # Phase 3: Additional random starts on sphere if time permits
    if best_points is not None and time.time() < deadline - 40:
        # Use best points as center for perturbation
        best_params = _points_to_sphere_params(best_points)
        rng = np.random.RandomState(12345)
        
        for attempt in range(6):
            if time.time() > deadline - 15:
                break
            
            # Perturb the best solution
            perturbation = rng.randn(n, 2) * 0.3
            perturbation[:, 0] = np.clip(perturbation[:, 0], -np.pi/2, np.pi/2)  # theta perturbation
            perturbation[:, 1] = perturbation[:, 1] % (2 * np.pi)  # wrap phi
            
            new_params = best_params + perturbation
            new_params[:, 0] = np.clip(new_params[:, 0], 0, np.pi)
            new_params[:, 1] = new_params[:, 1] % (2 * np.pi)
            
            try:
                params_polished = _polish_sphere(new_params, deadline)
                pts_polished = _points_to_sphere(params_polished)
                _update_best(pts_polished)
            except Exception:
                pass
    
    # Phase 4: Final high-precision Cartesian polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            final_points = _polish_cartesian(best_points, deadline)
            _update_best(final_points)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _make_icosahedron_plus_2()
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(best_points)):
        best_points = _make_icosahedron_plus_2()
    
    return best_points

# EVOLVE-BLOCK-END
