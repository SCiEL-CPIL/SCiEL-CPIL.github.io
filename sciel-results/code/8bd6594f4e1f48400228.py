"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
_mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[_mask].flatten()
IDX_J = IDX_J[_mask].flatten()
IDX_K = IDX_K[_mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert 14 parameters (7 angles + 7 radii) to 14 centrally symmetric points.
    params[0:7]: angles in (0, pi) for the 7 half-points
    params[7:14]: radii for the 7 half-points
    
    The 14 points are: p_i = r_i * (cos(a_i), sin(a_i)) and -p_i for i=0..6
    """
    angles = params[:N_HALF]
    radii = params[N_HALF:]
    
    angles = np.clip(angles, 0.02, np.pi - 0.02)
    radii = np.clip(radii, 0.05, 3.0)
    
    half_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    all_points = np.vstack([half_points, -half_points])
    return all_points

def _objective(params):
    """Negative score for minimization."""
    points = _params_to_points(params)
    score = _score(points)
    return -score

def _bounds_sym():
    """
    Bounds for centrally symmetric parameters.
    """
    bounds = []
    for _ in range(N_HALF):
        bounds.append((0.02, np.pi - 0.02))
    for _ in range(N_HALF):
        bounds.append((0.05, 3.0))
    return bounds

def _fourier_boundary(coeffs, n_samples=400):
    """
    Generate points on a smooth convex boundary from Fourier coefficients.
    coeffs: [a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    r(theta) = a0 + sum_{k=1}^{K} (ak*cos(k*theta) + bk*sin(k*theta))
    """
    K = (len(coeffs) - 1) // 2
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
    r = np.full(n_samples, coeffs[0])
    for k in range(1, K + 1):
        ak = coeffs[2 * k - 1]
        bk = coeffs[2 * k]
        r += ak * np.cos(k * thetas) + bk * np.sin(k * thetas)
    
    r_min = r.min()
    if r_min <= 0.1:
        r = r + (0.1 - r_min + 1e-6)
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return np.column_stack([x, y])

def _place_by_arc_length(boundary_pts, n_points, phase_shifts=None):
    """
    Place n_points at equal arc-length intervals along the boundary,
    with optional phase shifts.
    """
    n = len(boundary_pts)
    if phase_shifts is None:
        phase_shifts = np.zeros(n_points)
    
    diffs = np.diff(boundary_pts, axis=0, append=boundary_pts[:1])
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_length = np.cumsum(seg_lengths)
    total_length = cum_length[-1]
    
    target_fracs = (np.arange(n_points) + phase_shifts) / n_points
    target_fracs = target_fracs % 1.0
    
    points = np.zeros((n_points, 2))
    for i in range(n_points):
        target_len = target_fracs[i] * total_length
        idx = np.searchsorted(cum_length, target_len)
        if idx == 0:
            points[i] = boundary_pts[0]
        elif idx >= n:
            points[i] = boundary_pts[-1]
        else:
            seg_start = cum_length[idx - 1]
            seg_end = cum_length[idx]
            t = (target_len - seg_start) / (seg_end - seg_start + 1e-15)
            points[i] = (1 - t) * boundary_pts[idx - 1] + t * boundary_pts[idx]
    
    return points

def _optimize_fourier_coeffs(boundary_coeffs, n_samples=400, max_iter=200, lr=0.02):
    """
    Optimize Fourier coefficients to maximize the score.
    Uses coordinate descent with small perturbations.
    """
    best_coeffs = boundary_coeffs.copy()
    best_pts = _place_by_arc_length(_fourier_boundary(best_coeffs, n_samples), N)
    best_score = _score(best_pts)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(len(best_coeffs)):
            for delta in [-lr, -lr/2, lr/2, lr]:
                new_coeffs = best_coeffs.copy()
                new_coeffs[i] += delta
                boundary = _fourier_boundary(new_coeffs, n_samples)
                
                hull_area = _convex_hull_area(boundary)
                ref_hull = _convex_hull_area(np.column_stack(
                    [np.cos(np.linspace(0, 2*np.pi, n_samples, endpoint=False)),
                     np.sin(np.linspace(0, 2*np.pi, n_samples, endpoint=False))]))
                if hull_area < 0.85 * ref_hull:
                    continue
                
                pts = _place_by_arc_length(boundary, N)
                s = _score(pts)
                if s > best_score + 1e-10:
                    best_score = s
                    best_coeffs = new_coeffs
                    best_pts = pts
                    improved = True
        if not improved:
            break
    
    return best_coeffs, best_pts, best_score

def _optimize_phase_shifts(boundary_pts, n_points, initial_shifts, max_iter=300):
    """
    Optimize phase shifts to maximize the minimum triangle area.
    Uses coordinate descent.
    """
    best_shifts = initial_shifts.copy()
    best_pts = _place_by_arc_length(boundary_pts, n_points, best_shifts)
    best_score = _score(best_pts)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n_points):
            for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                new_shifts = best_shifts.copy()
                new_shifts[i] += delta
                new_shifts[i] = new_shifts[i] % 1.0
                pts = _place_by_arc_length(boundary_pts, n_points, new_shifts)
                s = _score(pts)
                if s > best_score + 1e-10:
                    best_score = s
                    best_shifts = new_shifts
                    best_pts = pts
                    improved = True
        if not improved:
            break
    
    return best_shifts, best_pts, best_score

def _optimize_ellipse_aspect(initial_pts, max_iter=50):
    """
    Optimize by scaling x and y coordinates independently.
    """
    best_pts = initial_pts.copy()
    best_score = _score(best_pts)
    
    for _ in range(max_iter):
        improved = False
        for sx in [0.9, 0.95, 1.0, 1.05, 1.1]:
            for sy in [0.9, 0.95, 1.0, 1.05, 1.1]:
                scaled = best_pts.copy()
                scaled[:, 0] *= sx
                scaled[:, 1] *= sy
                s = _score(scaled)
                if s > best_score + 1e-10:
                    best_score = s
                    best_pts = scaled
                    improved = True
        if not improved:
            break
    
    return best_pts, best_score

def _convex_hull_area(pts):
    """Compute convex hull area using Andrew's monotone chain."""
    pts = np.array(pts, dtype=float)
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples (vectorized)."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                                  (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if area < min_area:
                    min_area = area
    return min_area

def construct():
    """
    Construct 14 points maximizing min triangle area / convex hull area.
    Combines centrally symmetric parameterization with Fourier boundary + optimized point placement.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_points = None
    best_score = 0.0
    best_params = None
    rng = np.random.default_rng(42)
    
    bounds = _bounds_sym()
    
    # Generate initial parameter sets for local refinement
    inits = []
    
    # 1. Regular symmetric configuration (approximation of regular 14-gon)
    base_angles = np.linspace(0.1, np.pi - 0.1, N_HALF)
    base_radii = np.ones(N_HALF)
    inits.append(np.concatenate([base_angles, base_radii]))
    
    # 2. Varying radii scales
    for r_scale in [0.5, 0.7, 1.0, 1.3]:
        radii = np.full(N_HALF, r_scale)
        inits.append(np.concatenate([base_angles, radii]))
        
    # 3. Random perturbations around regular
    for _ in range(15):
        angles = np.sort(rng.uniform(0.1, np.pi - 0.1, N_HALF))
        radii = rng.uniform(0.5, 1.5, N_HALF)
        inits.append(np.concatenate([angles, radii]))
        
    # 4. More diverse random starts
    for _ in range(15):
        angles = np.sort(rng.uniform(0.05, np.pi - 0.05, N_HALF))
        radii = rng.uniform(0.2, 2.5, N_HALF)
        inits.append(np.concatenate([angles, radii]))

    # Phase 1: Global Exploration with Differential Evolution
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                polish=False,
                init='sobol',
                mutation=(0.3, 1.0),
                recombination=0.7
            )
            if np.isfinite(de_result.fun):
                score = -de_result.fun
                if score > best_score:
                    best_score = score
                    best_params = de_result.x.copy()
                    best_points = _params_to_points(de_result.x)
        except Exception:
            pass
    
    # Phase 2: Local Refinement from multiple starts
    scored_inits = []
    for init_params in inits:
        if time.time() > deadline - 40:
            break
        try:
            s = -_objective(init_params)
            scored_inits.append((s, init_params))
        except Exception:
            pass
            
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
        
    for s, p in scored_inits[:5]:
        candidates.append(p)
        
    regular_params = np.concatenate([np.linspace(0.1, np.pi - 0.1, N_HALF), np.ones(N_HALF)])
    candidates.append(regular_params)
    
    for init_params in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if np.isfinite(res.fun):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    best_points = _params_to_points(res.x)
                    
        except Exception:
            continue
        
        if best_score > 0 and time.time() < deadline - 5:
            try:
                res2 = minimize(
                    _objective,
                    best_params,
                    method='Powell',
                    options={
                        'maxiter': 500,
                        'xtol': 1e-12,
                        'ftol': 1e-14
                    }
                )
                score2 = -res2.fun
                if np.isfinite(score2) and score2 > best_score:
                    best_score = score2
                    best_params = res2.x.copy()
                    best_points = _params_to_points(res2.x)
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 2:
        try:
            res_final = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if np.isfinite(res_final.fun):
                score_final = -res_final.fun
                if score_final > best_score:
                    best_score = score_final
                    best_params = res_final.x.copy()
                    best_points = _params_to_points(res_final.x)
        except Exception:
            pass

    # Phase 4: Fourier boundary optimization (from current program)
    if time.time() < deadline - 20:
        n_samples = 400
        for n_harm in [1, 2, 3, 4]:
            for trial in range(8):
                if time.time() > deadline - 15:
                    break
                
                a0 = 1.0 + rng.uniform(-0.1, 0.1)
                coeffs = np.zeros(2 * (n_harm + 1))
                coeffs[0] = a0
                for k in range(1, n_harm + 1):
                    scale = 0.25 / k
                    coeffs[2 * k - 1] = rng.uniform(-scale, scale)
                    coeffs[2 * k] = rng.uniform(-scale, scale)
                
                boundary = _fourier_boundary(coeffs, n_samples)
                
                hull_area = _convex_hull_area(boundary)
                ref_hull = _convex_hull_area(np.column_stack(
                    [np.cos(np.linspace(0, 2*np.pi, n_samples, endpoint=False)),
                     np.sin(np.linspace(0, 2*np.pi, n_samples, endpoint=False))]))
                if hull_area < 0.85 * ref_hull:
                    continue
                
                initial_shifts = np.zeros(N)
                pts = _place_by_arc_length(boundary, N, initial_shifts)
                s = _score(pts)
                
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                
                if time.time() < deadline - 10:
                    opt_shifts, opt_pts, opt_score = _optimize_phase_shifts(
                        boundary, N, initial_shifts, max_iter=100
                    )
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
                
                if time.time() < deadline - 8:
                    opt_coeffs, opt_pts, opt_score = _optimize_fourier_coeffs(
                        coeffs, n_samples, max_iter=80, lr=0.015
                    )
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
                
                if time.time() < deadline - 5:
                    aspect_pts, aspect_score = _optimize_ellipse_aspect(best_points, max_iter=20)
                    if aspect_score > best_score:
                        best_score = aspect_score
                        best_points = aspect_pts.copy()

    # Phase 5: Multiple Nelder-Mead restarts from perturbed baselines
    if time.time() < deadline - 10:
        for trial in range(6):
            if time.time() > deadline - 5:
                break
            
            theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
            perturb = rng.normal(0, 0.08, (N, 2))
            pts = np.column_stack([np.cos(theta), np.sin(theta)]) + perturb
            
            def objective(x):
                p = x.reshape(N, 2)
                return -_score(p)
            
            x0 = pts.flatten()
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                
                nm_pts = result.x.reshape(N, 2)
                nm_score = -result.fun
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_pts.copy()
            except Exception:
                pass

    # Phase 6: Final fine-tuning with small perturbations
    if best_points is not None and time.time() < deadline - 3:
        for trial in range(50):
            if time.time() > deadline - 1:
                break
            perturbed = best_points + rng.normal(0, 0.02, (N, 2))
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_points = perturbed.copy()

    if best_points is None:
        theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(theta_reg), np.sin(theta_reg)])

    return best_points
