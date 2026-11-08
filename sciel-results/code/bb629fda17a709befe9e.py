N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

def _min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        if area < min_area:
            min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Compute all triangle areas as a vector."""
    n = len(points)
    areas = []
    for i, j, k in combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        areas.append(area)
    return np.array(areas)

def _soft_min_value(areas, temperature):
    """Log-sum-exp based soft-min of an array of values."""
    neg = -areas / temperature
    max_neg = np.max(neg)
    return -temperature * (max_neg + np.log(np.sum(np.exp(neg - max_neg))))

def _neg_score_flat(x):
    """Negative score for minimization (flat 2D points)."""
    points = x.reshape(N, 2)
    return -_score(points)

def _neg_soft_flat(x, temp):
    """Negative soft-min ratio for minimization (flat 2D points)."""
    points = x.reshape(N, 2)
    areas = _all_triangle_areas(points)
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 1.0
    soft_min = _soft_min_value(areas, temp)
    return -(soft_min / hull_area)

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular n-gon centered at center with given radius."""
    angles = 2 * np.pi * np.arange(n) / n
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _params_to_points(params):
    """
    Convert parameters to points.
    params: array of shape (2*N,) = [dr_0..dr_12, dtheta_0..dtheta_12]
    Point i: r_i = 1 + dr_i, theta_i = 2*pi*i/N + dtheta_i
    """
    base_angles = 2 * np.pi * np.arange(N) / N
    dr = params[:N]
    dtheta = params[N:]
    r = 1.0 + dr
    theta = base_angles + dtheta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _neg_score_params(params):
    """Negative score in parameter space."""
    points = _params_to_points(params)
    return -_score(points)

def _neg_soft_params(params, temp):
    """Negative soft-min ratio in parameter space."""
    points = _params_to_points(params)
    areas = _all_triangle_areas(points)
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 1.0
    soft_min = _soft_min_value(areas, temp)
    return -(soft_min / hull_area)

def _neg_soft_params_grad(params, temp):
    """Negative soft-min ratio in parameter space with gradient."""
    points = _params_to_points(params)
    areas = _all_triangle_areas(points)
    hull_area = _hull_area(points)
    if hull_area <= 1e-10:
        return 1.0, np.zeros(2*N)
    
    # Compute soft-min and its gradient w.r.t. areas
    neg = -areas / temp
    max_neg = np.max(neg)
    exp_neg = np.exp(neg - max_neg)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # softmax weights
    
    soft_min = -temp * (max_neg + np.log(sum_exp))
    
    # d(soft_min)/d(area_i) = -weights[i]
    # d(soft_min)/d(points) = sum_i d(soft_min)/d(area_i) * d(area_i)/d(points)
    # area_i = 0.5 * |cross|, so d(area_i)/d(points) involves cross products
    
    # For the ratio soft_min/hull_area, we need:
    # d(soft_min/hull_area)/d(points) = (d(soft_min)/d(points) * hull_area - soft_min * d(hull_area)/d(points)) / hull_area^2
    
    # This is complex. Let's use numerical gradient for now in the param space.
    # Actually, let's just use the function value and let L-BFGS-B do numerical gradients.
    
    return -(soft_min / hull_area), None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_score = -1.0
    best_points = None
    
    rng = np.random.RandomState(42)
    
    # Parameter bounds: dr in [-0.5, 0.5], dtheta in [-pi/13, pi/13]
    param_bounds = [(-0.5, 0.5)] * N + [(-np.pi/N, np.pi/N)] * N
    
    # Stage 1: Differential evolution with coarse soft-min
    if time.time() < deadline - 65:
        try:
            result = differential_evolution(
                _neg_soft_params,
                bounds=param_bounds,
                args=(0.1,),
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            opt_params = result.x
            opt_points = _params_to_points(opt_params)
            opt_score = _score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass
    
    # Stage 2: Differential evolution with medium soft-min
    if time.time() < deadline - 45 and best_points is not None:
        base_angles = 2 * np.pi * np.arange(N) / N
        r = np.sqrt(best_points[:, 0]**2 + best_points[:, 1]**2)
        theta = np.arctan2(best_points[:, 1], best_points[:, 0])
        dtheta = theta - base_angles
        dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
        dr = r - 1.0
        init_params = np.concatenate([dr, dtheta])
        
        try:
            result = differential_evolution(
                _neg_soft_params,
                bounds=param_bounds,
                args=(0.03,),
                seed=43,
                maxiter=60,
                popsize=12,
                tol=1e-7,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            opt_params = result.x
            opt_points = _params_to_points(opt_params)
            opt_score = _score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass
    
    # Stage 3: Fine-tune with Nelder-Mead on exact objective
    if time.time() < deadline - 20 and best_points is not None:
        try:
            result = minimize(
                _neg_score_flat,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            opt_pts = result.x.reshape(N, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Stage 4: Fine-tune with L-BFGS-B using fine soft-min
    if time.time() < deadline - 10 and best_points is not None:
        try:
            result = minimize(
                _neg_soft_flat,
                best_points.flatten(),
                method='L-BFGS-B',
                args=(0.005,),
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-12
                }
            )
            opt_pts = result.x.reshape(N, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Stage 5: Another Nelder-Mead polish
    if time.time() < deadline - 5 and best_points is not None:
        try:
            result = minimize(
                _neg_score_flat,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            opt_pts = result.x.reshape(N, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Stage 6: Final L-BFGS-B with very fine soft-min
    if time.time() < deadline - 1 and best_points is not None:
        try:
            result = minimize(
                _neg_soft_flat,
                best_points.flatten(),
                method='L-BFGS-B',
                args=(0.001,),
                options={
                    'maxiter': 200,
                    'ftol': 1e-15,
                    'gtol': 1e-13
                }
            )
            opt_pts = result.x.reshape(N, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0, center=(0,0))
    
    return best_points
# EVOLVE-BLOCK-END
