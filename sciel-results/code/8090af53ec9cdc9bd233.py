import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Pre-compute all triangle index combinations
_TRIS = list(combinations(range(N), 3))
_TRI_I = np.array([t[0] for t in _TRIS])
_TRI_J = np.array([t[1] for t in _TRIS])
_TRI_K = np.array([t[2] for t in _TRIS])
_N_TRI = len(_TRIS)  # 364

def _get_all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    pi = points[_TRI_I]
    pj = points[_TRI_J]
    pk = points[_TRI_K]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    return np.min(_get_all_triangle_areas(points))

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _normalize_points(points):
    """Center and scale points to unit hull area."""
    points = points - np.mean(points, axis=0)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return points
    scale = 1.0 / np.sqrt(hull_area)
    return points * scale

def _parametrized_objective(params):
    """
    Objective for parametrized optimization.
    params: [dtheta_0..dtheta_13, dr_0..dr_13]
    dtheta: angular perturbations (radians)
    dr: radial perturbations
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dtheta = params[:N]
    dr = params[N:]
    
    angles = base_angles + dtheta
    radii = 1.0 + dr
    
    # Ensure radii are positive
    if np.any(radii <= 0.1):
        return 1e6
    
    # Check for valid convex configuration
    # Points must be in convex position
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    # Check if all points are on the convex hull (strictly convex)
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            # Not all points are on the hull - penalty
            return 1e6
    except Exception:
        return 1e6
    
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    min_area = _min_triangle_area(points)
    return -min_area

def _smooth_neg_min_area(points_flat, tau=100.0):
    """Smooth surrogate using log-sum-exp."""
    points = points_flat.reshape(-1, 2)
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    areas = _get_all_triangle_areas(points)
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    smooth_val = -m + (1.0 / tau) * np.log(sum_exp)
    return smooth_val

def _local_refine(points, max_iters=50, step_size=0.005):
    """
    Local refinement that targets the worst triangles.
    Uses a more aggressive search with adaptive step sizes.
    """
    points = points.copy()
    current_score = _score(points)
    
    for it in range(max_iters):
        areas = _get_all_triangle_areas(points)
        min_area = np.min(areas)
        
        # Find triangles close to minimum
        threshold = min_area + 0.01 * min_area
        min_indices = np.where(areas <= threshold)[0]
        
        # Collect point indices involved in worst triangles
        worst_points = set()
        for tri_idx in min_indices[:20]:
            worst_points.add(_TRI_I[tri_idx])
            worst_points.add(_TRI_J[tri_idx])
            worst_points.add(_TRI_K[tri_idx])
        
        improved = False
        # Try multiple step sizes for each worst point
        for pt_idx in list(worst_points):
            for s_mult in [1.0, 0.5, 0.25, 2.0]:
                s = step_size * s_mult
                # Try 8 directions
                for dx in [-s, 0, s]:
                    for dy in [-s, 0, s]:
                        if dx == 0 and dy == 0:
                            continue
                        new_points = points.copy()
                        new_points[pt_idx, 0] += dx
                        new_points[pt_idx, 1] += dy
                        new_score = _score(new_points)
                        if new_score > current_score + 1e-12:
                            points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try all points with smaller step
            step_size *= 0.7
            if step_size < 1e-8:
                break
            improved = False
            for pt_idx in range(N):
                for dx in [-step_size, step_size]:
                    for dy in [-step_size, step_size]:
                        new_points = points.copy()
                        new_points[pt_idx, 0] += dx
                        new_points[pt_idx, 1] += dy
                        new_score = _score(new_points)
                        if new_score > current_score + 1e-12:
                            points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return points, current_score

def _fourier_objective(params, harmonics=3):
    """
    Objective for Fourier series parametrization.
    params: [a_1, b_1, a_2, b_2, ..., a_h, b_h, dtheta_0..dtheta_13]
    where r(theta) = 1 + sum_{k=1}^{h} (a_k cos(k theta) + b_k sin(k theta))
    """
    h = harmonics
    n_fourier = 2 * h
    fourier_coeffs = params[:n_fourier]
    dtheta = params[n_fourier:]
    
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + dtheta
    
    # Compute radius for each angle
    radii = np.ones(N)
    for k in range(1, h + 1):
        a_k = fourier_coeffs[2 * (k - 1)]
        b_k = fourier_coeffs[2 * (k - 1) + 1]
        radii += a_k * np.cos(k * angles) + b_k * np.sin(k * angles)
    
    # Ensure radii are positive and bounded
    if np.any(radii <= 0.5) or np.any(radii >= 2.0):
        return 1e6
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    # Check if all points are on the convex hull
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            return 1e6
    except Exception:
        return 1e6
    
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    min_area = _min_triangle_area(points)
    return -min_area

def _fourier_smooth_objective(params, harmonics=3, tau=100.0):
    """Smooth version of Fourier objective."""
    h = harmonics
    n_fourier = 2 * h
    fourier_coeffs = params[:n_fourier]
    dtheta = params[n_fourier:]
    
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + dtheta
    
    radii = np.ones(N)
    for k in range(1, h + 1):
        a_k = fourier_coeffs[2 * (k - 1)]
        b_k = fourier_coeffs[2 * (k - 1) + 1]
        radii += a_k * np.cos(k * angles) + b_k * np.sin(k * angles)
    
    if np.any(radii <= 0.5) or np.any(radii >= 2.0):
        return 1e6
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            return 1e6
    except Exception:
        return 1e6
    
    points = _normalize_points(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    areas = _get_all_triangle_areas(points)
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    smooth_val = -m + (1.0 / tau) * np.log(sum_exp)
    return smooth_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    def _update_best(points):
        nonlocal best_score, best_points
        points = _normalize_points(points)
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate initial configurations
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(x0_reg)
    
    # 2. Regular 14-gon with noise
    configs.append(x0_reg + 0.05 * np.random.randn(N, 2))
    
    # 3. Slightly distorted regular 14-gon (Fourier with small coeffs)
    for _ in range(3):
        params = np.random.randn(2 * 3 + N) * 0.1
        h = 3
        n_fourier = 2 * h
        fourier_coeffs = params[:n_fourier]
        dtheta = params[n_fourier:]
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles_f = base_angles + dtheta
        radii = np.ones(N)
        for k in range(1, h + 1):
            a_k = fourier_coeffs[2 * (k - 1)]
            b_k = fourier_coeffs[2 * (k - 1) + 1]
            radii += a_k * np.cos(k * angles_f) + b_k * np.sin(k * angles_f)
        x = radii * np.cos(angles_f)
        y = radii * np.sin(angles_f)
        configs.append(np.column_stack([x, y]))
    
    # Evaluate initial configs
    for config in configs:
        _update_best(config)
    
    # Differential Evolution on parametrized space (angular + radial)
    if time.time() < deadline - 30:
        try:
            bounds = [(-0.2, 0.2)] * N + [(-0.2, 0.2)] * N
            result_de = differential_evolution(
                _parametrized_objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False
            )
            if result_de.fun < 1e5:
                params_opt = result_de.x
                base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                dtheta = params_opt[:N]
                dr = params_opt[N:]
                angles_p = base_angles + dtheta
                radii_p = 1.0 + dr
                x_p = radii_p * np.cos(angles_p)
                y_p = radii_p * np.sin(angles_p)
                points_p = np.column_stack([x_p, y_p])
                _update_best(points_p)
        except Exception:
            pass
    
    # Differential Evolution on Fourier space
    if time.time() < deadline - 30:
        try:
            h = 3
            n_fourier = 2 * h
            bounds_f = [(-0.1, 0.1)] * n_fourier + [(-0.15, 0.15)] * N
            result_de_f = differential_evolution(
                _fourier_objective,
                bounds_f,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=43,
                polish=False,
                args=(h,)
            )
            if result_de_f.fun < 1e5:
                params_f = result_de_f.x
                fourier_coeffs = params_f[:n_fourier]
                dtheta_f = params_f[n_fourier:]
                base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                angles_ff = base_angles + dtheta_f
                radii_ff = np.ones(N)
                for k in range(1, h + 1):
                    a_k = fourier_coeffs[2 * (k - 1)]
                    b_k = fourier_coeffs[2 * (k - 1) + 1]
                    radii_ff += a_k * np.cos(k * angles_ff) + b_k * np.sin(k * angles_ff)
                x_ff = radii_ff * np.cos(angles_ff)
                y_ff = radii_ff * np.sin(angles_ff)
                points_ff = np.column_stack([x_ff, y_ff])
                _update_best(points_ff)
        except Exception:
            pass
    
    # Local refinement on the best point found so far
    if best_points is not None and time.time() < deadline - 10:
        refined_points, refined_score = _local_refine(best_points, max_iters=30, step_size=0.005)
        _update_best(refined_points)
        
        # One more round of refinement with smaller steps
        if time.time() < deadline - 5:
            refined_points2, refined_score2 = _local_refine(best_points, max_iters=20, step_size=0.001)
            _update_best(refined_points2)
    
    # Final local refinement with the best found
    if best_points is not None and time.time() < deadline - 2:
        final_points, final_score = _local_refine(best_points, max_iters=10, step_size=0.0005)
        _update_best(final_points)
    
    # If still nothing, return regular polygon
    if best_points is None:
        best_points = x0_reg
    
    return best_points
