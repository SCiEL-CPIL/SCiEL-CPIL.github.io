"""
Bootstrap spawn: MODEL-AND-OPTIMIZE approach.

Parameterize the 14 points in a structured way (e.g., radii and angles for a
convex-positioned set, or a two-ring model) and fit the parameters directly
against the task's objective: maximize min_triangle_area / hull_area.

The key insight for "maximize the minimum triangle area" with N points in a
convex region is that an optimal configuration tends to be close to a regular
polygon or a perturbation thereof. We use a parameterized model:

  - Outer ring: K points on an ellipse or perturbed circle (these define the hull)
  - Inner ring: (N-K) points at smaller radii

Then we optimize the parameters (radii, angles, possibly elliptic eccentricity)
directly.

For N=14, a good structure is: 10 points on an outer ring (roughly decagonal)
and 4 points on an inner ring. This balances the minimum triangle area.

We use scipy.optimize with a smooth surrogate (soft-min of triangle areas)
combined with the exact score for final evaluation.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 143.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3 = 364)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _soft_obj(pts_flat, beta=100.0):
    """Soft objective: hull_area / soft_min(triangle areas)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _soft_obj_grad(pts_flat, beta=100.0):
    """Soft objective with numerical gradient (for L-BFGS-B)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10, None
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10, None
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10, None
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10, None
    return hull_area / soft_min_val, None


def _two_ring_params_to_pts(outer_n, inner_n, params):
    """
    Convert two-ring model parameters to point coordinates.
    
    params layout:
      [0:outer_n]   outer angles (radians)
      [outer_n:outer_n+inner_n] inner angles (radians)
      [outer_n+inner_n]         outer radius scale
      [outer_n+inner_n+1]       inner radius scale
      [outer_n+inner_n+2]       elliptic x-scale
      [outer_n+inner_n+3]       elliptic y-scale
      [outer_n+inner_n+4]       outer angular perturbation amplitude
      [outer_n+inner_n+5]       inner angular perturbation amplitude
    """
    outer_angles = params[0:outer_n]
    inner_angles = params[outer_n:outer_n+inner_n]
    outer_r = params[outer_n+inner_n]
    inner_r = params[outer_n+inner_n+1]
    ex = params[outer_n+inner_n+2]
    ey = params[outer_n+inner_n+3]
    outer_pert = params[outer_n+inner_n+4]
    inner_pert = params[outer_n+inner_n+5]
    
    n_total = outer_n + inner_n
    base_outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n
    base_inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    
    outer_angles_actual = base_outer_angles + outer_pert * np.sin(2 * base_outer_angles) + outer_angles
    inner_angles_actual = base_inner_angles + inner_pert * np.sin(3 * base_inner_angles) + inner_angles
    
    outer_pts = np.column_stack([
        outer_r * ex * np.cos(outer_angles_actual),
        outer_r * ey * np.sin(outer_angles_actual)
    ])
    inner_pts = np.column_stack([
        inner_r * ex * np.cos(inner_angles_actual),
        inner_r * ey * np.sin(inner_angles_actual)
    ])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts


def _optimize_two_ring(time_budget):
    """
    Optimize a two-ring model: 10 outer + 4 inner points.
    Parameters are angles, radii, elliptic scales, and perturbation amplitudes.
    """
    start_time = time.time()
    
    outer_n = 10
    inner_n = 4
    
    def obj_func(params):
        if time.time() - start_time > time_budget:
            raise TimeoutError()
        pts = _two_ring_params_to_pts(outer_n, inner_n, params)
        val, grad = _soft_obj_grad(pts.flatten(), beta=100.0)
        return val, grad
    
    # Initial parameters: regular polygon configuration
    outer_angles = np.zeros(outer_n)
    inner_angles = np.zeros(inner_n)
    outer_r = 1.0
    inner_r = 0.5
    ex = 1.0
    ey = 1.0
    outer_pert = 0.0
    inner_pert = 0.0
    
    x0 = np.concatenate([outer_angles, inner_angles,
                          [outer_r, inner_r, ex, ey, outer_pert, inner_pert]])
    
    best_params = x0.copy()
    best_val = _soft_obj(pts_to_flat(_two_ring_params_to_pts(outer_n, inner_n, x0)), beta=100.0)
    
    # Try multiple beta values
    for beta in [50, 100, 200, 400]:
        if time.time() - start_time > time_budget - 2:
            break
        
        def obj_beta(params):
            pts = _two_ring_params_to_pts(outer_n, inner_n, params)
            return _soft_obj(pts.flatten(), beta=beta)
        
        try:
            res = minimize(obj_beta, best_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                                  'adaptive': True, 'maxfev': 5000})
            if res.x is not None:
                val = _soft_obj(_two_ring_params_to_pts(outer_n, inner_n, res.x).flatten(), beta=beta)
                if val < best_val:
                    best_val = val
                    best_params = res.x
        except Exception:
            pass
    
    pts = _two_ring_params_to_pts(outer_n, inner_n, best_params)
    return pts


def pts_to_flat(pts):
    return pts.flatten()


def _optimize_full_points(time_budget, init_pts):
    """
    Optimize all 28 coordinates directly with Nelder-Mead on the soft objective.
    """
    start_time = time.time()
    x0 = init_pts.flatten()
    
    best_x = x0.copy()
    best_score = _score(init_pts)
    
    # Stage 1: Soft objective with increasing beta
    for beta in [30, 60, 120, 250, 500]:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj_func(x):
            return _soft_obj(x, beta=beta)
        
        try:
            res = minimize(obj_func, best_x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                                  'adaptive': True, 'maxfev': 4000})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_x = res.x
        except Exception:
            pass
    
    # Stage 2: Exact score polishing
    if time.time() - start_time < time_budget - 2:
        try:
            res = minimize(_neg_score, best_x, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True, 'maxfev': 8000})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_x = res.x
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_two_ring(n):
    """Two-ring initialization: 10 outer + 4 inner."""
    outer_n = 10
    inner_n = n - outer_n
    outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_perturbed_polygon(n, seed=42):
    """Regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    angles = 2 * np.pi * np.arange(n) / n + rng.normal(0, 0.05, n)
    radii = 1.0 + rng.normal(0, 0.05, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_golden_spiral(n, seed=42):
    """Points on a golden-angle spiral, giving good coverage."""
    rng = np.random.RandomState(seed)
    golden_angle = np.pi * (3 - np.sqrt(5))
    angles = np.arange(n) * golden_angle
    radii = np.linspace(0.3, 1.0, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed_rings(n, seed=42):
    """Three-ring initialization: 7 outer, 4 middle, 3 inner."""
    outer_n, mid_n, inner_n = 7, 4, 3
    outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n
    mid_angles = 2 * np.pi * np.arange(mid_n) / mid_n + np.pi / mid_n
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    mid = 0.65 * np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])
    inner = 0.35 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, mid, inner])
    return _normalize_unit_area(pts)


def _polar_to_cartesian(params):
    """
    Convert polar parameters to Cartesian points with enforced angular ordering.
    params: [r_0, ..., r_{N-1}, theta_0, d_theta_0, ..., d_theta_{N-2}]
    Total: 2N - 1 parameters.
    theta_k = theta_0 + cumsum(d_theta_0..d_theta_{k-1})
    We enforce d_theta_k > 0 so angles are strictly increasing.
    """
    n = N
    radii = np.abs(params[:n]) + 0.01  # ensure positive
    theta_start = params[n]
    d_thetas = np.abs(params[n+1:n+n-1]) + 0.05  # ensure positive gaps
    thetas = np.zeros(n)
    thetas[0] = theta_start
    thetas[1:] = theta_start + np.cumsum(d_thetas)
    # Normalize: subtract mean to center
    thetas = thetas - np.mean(thetas)
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.column_stack([x, y])


def _neg_score_polar(params):
    """Negative score in polar parameter space."""
    pts = _polar_to_cartesian(params)
    s = _score(pts)
    return -s


def _global_search_polar(time_budget):
    """
    Global search using differential evolution in polar coordinate space.
    Returns the best Cartesian points found.
    """
    start_time = time.time()
    
    n = N
    # params: [r_0..r_{n-1}, theta_0, d_theta_0..d_theta_{n-2}]
    n_params = 2 * n - 1
    
    # Bounds: radii in [0.05, 2.0], theta_start in [0, 2*pi], d_theta in [0.1, 2*pi]
    bounds = [(0.05, 2.0)] * n + [(0, 2*np.pi)] + [(0.1, 2*np.pi)] * (n - 1)
    
    def de_obj(params):
        if time.time() - start_time > time_budget:
            return 1e10
        pts = _polar_to_cartesian(params)
        s = _score(pts)
        return -s
    
    try:
        res = differential_evolution(
            de_obj, bounds,
            seed=42, maxiter=200, popsize=15,
            tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
            polish=False, init='sobol',
            updating='immediate',
            callback=None
        )
        if res.x is not None:
            pts = _polar_to_cartesian(res.x)
            return pts
    except Exception:
        pass
    
    return None


def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_two_ring(N),
        _initial_perturbed_polygon(N, seed=1),
        _initial_perturbed_polygon(N, seed=2),
        _initial_perturbed_polygon(N, seed=3),
        _initial_golden_spiral(N, seed=42),
        _initial_mixed_rings(N, seed=42),
    ]
    
    # Evaluate initial scores
    init_scores = []
    for init in inits:
        s = _score(init)
        init_scores.append(s)
    
    # Sort by initial score (best first)
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    init_scores = [init_scores[i] for i in sorted_indices]
    
    best_pts = inits[0]
    best_score = init_scores[0]
    
    # Time allocation
    total_time = DEADLINE - (time.time() - start_time)
    de_time = total_time * 0.20
    two_ring_time = total_time * 0.15
    full_opt_time = total_time * 0.50
    polish_time = total_time * 0.15
    
    # Phase 1: Global search via differential evolution in polar space
    if time.time() - start_time < DEADLINE - two_ring_time - full_opt_time - polish_time - 5:
        try:
            de_pts = _global_search_polar(de_time * 0.9)
            if de_pts is not None:
                s = _score(de_pts)
                if s > best_score:
                    best_score = s
                    best_pts = de_pts
        except Exception:
            pass
    
    # Phase 2: Structured two-ring model
    if time.time() - start_time < DEADLINE - full_opt_time - polish_time - 5:
        try:
            tr_pts = _optimize_two_ring(two_ring_time * 0.8)
            s = _score(tr_pts)
            if s > best_score:
                best_score = s
                best_pts = tr_pts
        except Exception:
            pass
    
    # Phase 3: Full-point optimization from top initializations
    remaining = DEADLINE - polish_time - 5 - (time.time() - start_time)
    
    # Include best_pts as an additional init
    all_inits = [best_pts] + inits[:4]
    # Deduplicate by score proximity
    unique_inits = [all_inits[0]]
    for init in all_inits[1:]:
        s = _score(init)
        if s > best_score - 0.01:
            unique_inits.append(init)
    
    for idx, init in enumerate(unique_inits[:4]):
        if time.time() - start_time > DEADLINE - polish_time - 5:
            break
        
        time_left = DEADLINE - polish_time - 5 - (time.time() - start_time)
        if time_left < 5:
            break
        
        try:
            cand = _optimize_full_points(time_left * 0.9, init)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass
    
    # Phase 4: High-precision polish
    if time.time() - start_time < DEADLINE - 3:
        try:
            polish_budget = min(polish_time, DEADLINE - 5 - (time.time() - start_time))
            if polish_budget > 2:
                # Stage 4a: Soft-min with high beta
                def final_obj_high_beta(x):
                    return _soft_obj(x, beta=1000.0)
                
                res = minimize(final_obj_high_beta, best_pts.flatten(), method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13,
                                      'adaptive': True, 'maxfev': 6000})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
                
                # Stage 4b: Exact score polish with very tight tolerance
                if time.time() - start_time < DEADLINE - 3:
                    res2 = minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                          'adaptive': True, 'maxfev': 10000})
                    if res2.x is not None:
                        cand = res2.x.reshape(-1, 2)
                        s = _score(cand)
                        if s > best_score:
                            best_score = s
                            best_pts = cand
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
