import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

# Precompute all triangle index triples for vectorized area calculation
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _normalize(points):
    """Normalize points to have hull area 1 and center at origin."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return points
    scale = 1.0 / np.sqrt(hull_area)
    center = np.mean(points, axis=0)
    return (points - center) * scale


def _points_from_polar(angles, radii):
    """
    Convert sorted angular coordinates and radii to Cartesian points.
    angles: array of shape (N,) with values in [0, 2*pi), sorted
    radii: array of shape (N,) with positive values
    Returns points of shape (N, 2).
    """
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _polar_barrier_objective(params, target_ratio, mu, deadline=None):
    """
    Log-barrier constrained optimization in polar coordinates.
    
    params: array of shape (2*N,) = [angles (N), log_radii (N)]
    target_ratio: target minimum triangle area / hull area ratio
    mu: barrier parameter (increases over iterations for tighter constraint)
    
    The objective minimizes:
      -log(min_area / hull_area) + mu * sum(max(0, target_ratio - area_i/hull_area)^2)
    
    This is a smooth approximation that encourages all triangle area ratios
    to be at least target_ratio, while maximizing the minimum ratio.
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    angles = params[:N]
    log_radii = params[N:]
    radii = np.exp(log_radii)
    
    # Ensure angles are in [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    points = _points_from_polar(angles, radii)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    ratios = areas / hull_area
    
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    # Main term: maximize min ratio (minimize negative log)
    main_term = -np.log(r_min)
    
    # Barrier term: penalize ratios below target
    violations = np.maximum(0.0, target_ratio - ratios)
    barrier_term = mu * np.sum(violations ** 2)
    
    return main_term + barrier_term


def _polar_soft_min_objective(params, beta=50.0, deadline=None):
    """
    Soft-min surrogate in polar coordinates for differentiable optimization.
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    angles = params[:N]
    log_radii = params[N:]
    radii = np.exp(log_radii)
    
    angles = np.mod(angles, 2 * np.pi)
    points = _points_from_polar(angles, radii)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    ratios = areas / hull_area
    
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _cartesian_soft_min_objective(x_flat, beta=50.0, deadline=None):
    """Soft-min surrogate in Cartesian coordinates."""
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    ratios = areas / hull_area
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _cartesian_hard_objective(x_flat, deadline=None):
    """Hard objective in Cartesian: -min_triangle_area / hull_area"""
    if deadline is not None and time.time() > deadline:
        return 1e10
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _polar_to_cartesian_params(angles, log_radii):
    """Convert polar params to Cartesian flat array."""
    radii = np.exp(log_radii)
    angles = np.mod(angles, 2 * np.pi)
    points = _points_from_polar(angles, radii)
    return points.flatten()


def _cartesian_to_polar_params(points):
    """Convert Cartesian points to polar params [angles, log_radii]."""
    x = points[:, 0]
    y = points[:, 1]
    radii = np.sqrt(x**2 + y**2)
    radii = np.maximum(radii, 1e-8)
    angles = np.arctan2(y, x)
    angles = np.mod(angles, 2 * np.pi)
    log_radii = np.log(radii)
    return np.concatenate([angles, log_radii])


def _optimize_polar_barrier(init_params, deadline, target_ratio=0.03, mu_start=1.0, mu_end=100.0, n_iter=100):
    """
    Optimize in polar coordinates using log-barrier method.
    Gradually increases the barrier parameter mu.
    """
    current_params = init_params.copy()
    
    # Warm-start: try to get a reasonable initial score
    angles = np.mod(current_params[:N], 2 * np.pi)
    log_radii = current_params[N:]
    radii = np.exp(log_radii)
    points = _points_from_polar(angles, radii)
    current_score = _score(points)
    
    # Set target slightly above current score if possible
    if current_score > 0:
        target_ratio = max(target_ratio, current_score * 1.1)
    
    for it in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        # Linearly interpolate mu
        t = it / max(1, n_iter - 1)
        mu = mu_start * (mu_end / mu_start) ** t
        
        # Run a short optimization with current mu
        try:
            result = minimize(
                _polar_barrier_objective,
                current_params,
                args=(target_ratio, mu, deadline),
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result is not None:
                current_params = result.x
        except Exception:
            pass
        
        # Evaluate current score
        angles = np.mod(current_params[:N], 2 * np.pi)
        log_radii = current_params[N:]
        radii = np.exp(log_radii)
        points = _points_from_polar(angles, radii)
        score = _score(points)
        
        # Adapt target if we're doing well
        if score > target_ratio * 1.2:
            target_ratio = min(score * 1.05, 0.15)
    
    # Final evaluation
    angles = np.mod(current_params[:N], 2 * np.pi)
    log_radii = current_params[N:]
    radii = np.exp(log_radii)
    points = _points_from_polar(angles, radii)
    final_score = _score(points)
    
    return points, final_score


def _optimize_polar_softmin(init_params, deadline, max_iter=200):
    """Optimize in polar coordinates using soft-min surrogate."""
    try:
        result = minimize(
            _polar_soft_min_objective,
            init_params,
            args=(50.0, deadline),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if result is not None:
            angles = np.mod(result.x[:N], 2 * np.pi)
            log_radii = result.x[N:]
            radii = np.exp(log_radii)
            points = _points_from_polar(angles, radii)
            score = _score(points)
            return points, score
    except Exception:
        pass
    return None, 0.0


def _optimize_cartesian_softmin(init_points, deadline, max_iter=200):
    """Optimize in Cartesian coordinates using soft-min surrogate."""
    try:
        result = minimize(
            _cartesian_soft_min_objective,
            init_points.flatten(),
            args=(50.0, deadline),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if result is not None:
            points = result.x.reshape(N, 2)
            points = _normalize(points)
            score = _score(points)
            return points, score
    except Exception:
        pass
    return None, 0.0


def _optimize_cartesian_hard(init_points, deadline, max_iter=200):
    """Optimize in Cartesian coordinates using hard objective."""
    try:
        result = minimize(
            _cartesian_hard_objective,
            init_points.flatten(),
            args=(deadline,),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if result is not None:
            points = result.x.reshape(N, 2)
            points = _normalize(points)
            score = _score(points)
            return points, score
    except Exception:
        pass
    return None, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initial configurations in polar form
    # Config 1: Regular 13-gon
    reg_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_log_radii = np.zeros(N)
    reg_params = np.concatenate([reg_angles, reg_log_radii])
    
    # Config 2: Perturbed regular polygon
    pert_angles = reg_angles + rng.normal(0, 0.05, size=N)
    pert_log_radii = rng.normal(0, 0.1, size=N)
    pert_params = np.concatenate([pert_angles, pert_log_radii])
    
    # Config 3: Two-ring structure (inner 5 + outer 8)
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    all_angles = np.sort(np.concatenate([inner_angles, outer_angles]))
    inner_radii = np.full(5, 0.5)
    outer_radii = np.full(8, 1.0)
    # Need to assign radii in sorted angle order
    sorted_angles = all_angles
    # Determine which are inner/outer based on angle
    two_ring_log_radii = np.zeros(N)
    for i, ang in enumerate(sorted_angles):
        # Check if this angle is close to an inner angle
        min_diff = np.min(np.abs(ang - inner_angles))
        if min_diff < 0.1:
            two_ring_log_radii[i] = np.log(0.5)
        else:
            two_ring_log_radii[i] = 0.0
    two_ring_params = np.concatenate([sorted_angles, two_ring_log_radii])
    
    # Config 4: Random convex configuration
    rand_angles = np.sort(rng.uniform(0, 2 * np.pi, N))
    rand_log_radii = rng.uniform(-0.5, 0.5, N)
    rand_params = np.concatenate([rand_angles, rand_log_radii])
    
    polar_configs = [reg_params, pert_params, two_ring_params, rand_params]
    
    # Phase 2: Polar barrier optimization for each config
    for cfg_idx, init_params in enumerate(polar_configs):
        if time.time() > deadline - 30:
            break
        
        points, score = _optimize_polar_barrier(init_params, deadline)
        if points is not None:
            points = _normalize(points)
            score = _score(points)
            update_best(points, score)
    
    # Phase 3: Polar soft-min refinement from best
    if best_points is not None and time.time() < deadline - 20:
        best_polar = _cartesian_to_polar_params(best_points)
        points, score = _optimize_polar_softmin(best_polar, deadline, max_iter=150)
        if points is not None:
            points = _normalize(points)
            score = _score(points)
            update_best(points, score)
    
    # Phase 4: Cartesian soft-min from best
    if best_points is not None and time.time() < deadline - 15:
        points, score = _optimize_cartesian_softmin(best_points, deadline, max_iter=150)
        if points is not None:
            update_best(points, score)
    
    # Phase 5: Cartesian hard refinement
    if best_points is not None and time.time() < deadline - 10:
        points, score = _optimize_cartesian_hard(best_points, deadline, max_iter=200)
        if points is not None:
            update_best(points, score)
    
    # Phase 6: Additional random restarts if time permits
    if time.time() < deadline - 25:
        for i in range(2):
            if time.time() > deadline - 15:
                break
            
            # Random polar config
            rand_ang = np.sort(rng.uniform(0, 2 * np.pi, N))
            rand_lr = rng.uniform(-0.3, 0.3, N)
            rand_polar = np.concatenate([rand_ang, rand_lr])
            
            points, score = _optimize_polar_barrier(rand_polar, deadline, target_ratio=0.02)
            if points is not None:
                points = _normalize(points)
                score = _score(points)
                if update_best(points, score):
                    # Refine
                    if time.time() < deadline - 10:
                        points2, score2 = _optimize_cartesian_hard(best_points, deadline, max_iter=100)
                        if points2 is not None:
                            update_best(points2, score2)
    
    # Final fine-tuning
    if best_points is not None and time.time() < deadline - 5:
        points, score = _optimize_cartesian_hard(best_points, deadline, max_iter=300)
        if points is not None:
            update_best(points, score)
    
    if best_points is None:
        return _regular_polygon(N)
    
    return best_points

# EVOLVE-BLOCK-END
