import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 13
DEADLINE = 140.0  # safety margin before 150s kill

def _min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score = min triangle area / convex hull area."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _score_neg(points):
    return -_score(points)

def _logsumexp_surrogate(points, beta=50.0):
    """
    Smooth surrogate: -logsumexp(-beta * areas) / hull_area
    This is a smooth approximation of min_area / hull_area.
    Returns negative for minimization.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute all triangle areas vectorized
    # C(13,3) = 286 triangles
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    
    if len(areas) == 0:
        return 0.0
    
    # log-sum-exp of -beta * areas gives smooth min
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    smooth_min = -lse / beta
    
    # Hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    return -(smooth_min / hull_area)

def _logsumexp_surrogate_grad(points, beta=50.0):
    """
    Gradient of the log-sum-exp surrogate for L-BFGS-B.
    Returns (value, gradient) where value is negative score (for minimization).
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(points)
    
    # Precompute all triangle indices
    triples = list(itertools.combinations(range(n), 3))
    
    # Compute areas
    areas = np.zeros(len(triples))
    for idx, (i, j, k) in enumerate(triples):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    
    # Hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return (0.0, np.zeros(2 * n))
    except Exception:
        return (0.0, np.zeros(2 * n))
    
    # Log-sum-exp
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_vals = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_vals)
    lse = max_val + np.log(sum_exp)
    smooth_min = -lse / beta
    
    # Probabilities (softmax weights)
    probs = exp_vals / sum_exp
    
    # Value
    value = -(smooth_min / hull_area)
    
    # Gradient computation
    grad = np.zeros(2 * n)
    
    # Gradient of smooth_min w.r.t. points
    # d(smooth_min)/d(area_m) = -probs[m] / 1  (since smooth_min = -lse/beta, d lse/d area_m = -beta * probs[m])
    # Actually: lse = max_val + log(sum(exp(-beta*areas)))
    # d lse / d areas[m] = -beta * exp(-beta*areas[m]) / sum_exp = -beta * probs[m]
    # smooth_min = -lse / beta
    # d smooth_min / d areas[m] = probs[m]
    
    # Now d areas[m] / d points
    for m, (i, j, k) in enumerate(triples):
        if probs[m] < 1e-15:
            continue
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        # area = 0.5 * |cross|, assume cross > 0 for gradient (use sign)
        sign = 1.0 if cross >= 0 else -1.0
        
        # d area / d points[i] = -0.5 * sign * [v2[1], -v2[0]]
        # d area / d points[j] =  0.5 * sign * [-v2[1], v2[0]]  ... let me be more careful
        # area = 0.5 * |v1_x * v2_y - v1_y * v2_x|
        # Let f = v1_x * v2_y - v1_y * v2_x
        # v1 = p_j - p_i, v2 = p_k - p_i
        # f = (p_jx - p_ix)(p_ky - p_iy) - (p_jy - p_iy)(p_kx - p_ix)
        # df/dp_ix = -(p_ky - p_iy) + (p_jy - p_iy) = -v2[1] + v1[1]
        # df/dp_iy = (p_jx - p_ix) - (p_kx - p_ix) = v1[0] - v2[0]
        # df/dp_jx = (p_ky - p_iy) = v2[1]
        # df/dp_jy = -(p_kx - p_ix) = -v2[0]
        # df/dp_kx = -(p_jy - p_iy) = -v1[1]
        # df/dp_ky = (p_jx - p_ix) = v1[0]
        
        df_di = np.array([-v2[1] + v1[1], v1[0] - v2[0]])
        df_dj = np.array([v2[1], -v2[0]])
        df_dk = np.array([-v1[1], v1[0]])
        
        w = 0.5 * sign * probs[m]
        
        grad[2*i:2*i+2] += -w * df_di
        grad[2*j:2*j+2] += w * df_dj
        grad[2*k:2*k+2] += w * df_dk
    
    # d value / d points = -1/hull_area * d smooth_min / d points
    # (ignoring derivative of hull_area for speed, or we could include it)
    grad = -grad / hull_area
    
    return (value, grad)

def _elliptical_13gon(theta, a, b, c):
    """
    Generate 13 points as a regular 13-gon distorted by elliptical quadratic form.
    Q = [[a, c], [c, b]] must be PSD.
    Points: p_i = Q @ regular_13gon_vertex_i
    """
    angles = 2 * np.pi * np.arange(13) / 13 + theta
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    Q = np.array([[a, c], [c, b]])
    return Q @ regular.T

def _psd_check(a, b, c):
    """Check if [[a,c],[c,b]] is PSD: a>=0, b>=0, ab-c^2>=0."""
    if a < -1e-10 or b < -1e-10:
        return False
    if a * b - c * c < -1e-10:
        return False
    return True

def construct():
    t0 = time.time()
    best_points = None
    best_score = 0.0
    
    def _update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s
    
    # Phase 1: Optimize elliptical 13-gon parameters (theta, a, b, c)
    # Parameterize PSD matrix: a = u^2, b = v^2, c = u*v*w where |w|<=1
    # Or use: a = exp(x1), b = exp(x2), c = exp(x3) with constraint
    # Simpler: use Cholesky-like: Q = R^T R where R = [[r11, r12], [0, r22]]
    # Q = [[r11^2, r11*r12], [r11*r12, r11^2*r12^2 + r22^2]]
    
    def _elliptic_params_to_points(params):
        theta, r11, r12, r22 = params
        if r11 < 0.01: r11 = 0.01
        if r22 < 0.01: r22 = 0.01
        Q = np.array([[r11**2, r11*r12], [r11*r12, r11**2*r12**2 + r22**2]])
        angles = 2 * np.pi * np.arange(13) / 13 + theta
        regular = np.column_stack([np.cos(angles), np.sin(angles)])
        return Q @ regular.T
    
    def _elliptic_obj(params):
        pts = _elliptic_params_to_points(params)
        return -_score(pts)
    
    # Try several initial elliptical configurations
    elliptic_inits = [
        [0.0, 1.0, 0.0, 1.0],   # circle
        [0.0, 1.5, 0.0, 0.8],   # ellipse
        [0.2, 1.2, 0.3, 1.0],   # rotated ellipse
        [0.5, 0.8, 0.2, 1.3],   # another
        [0.0, 1.0, 0.5, 1.0],   # skewed
    ]
    
    for init in elliptic_inits:
        if time.time() - t0 > DEADLINE - 30:
            break
        try:
            res = minimize(_elliptic_obj, init, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10})
            pts = _elliptic_params_to_points(res.x)
            _update_best(pts)
        except Exception:
            pass
    
    # Phase 2: Free 2D optimization from best elliptic result
    if best_points is not None:
        if time.time() - t0 < DEADLINE - 40:
            # Use log-sum-exp surrogate with gradient
            def _surrogate_obj(params):
                pts = params.reshape(-1, 2)
                val, _ = _logsumexp_surrogate_grad(pts, beta=50.0)
                return val
            
            x0 = best_points.flatten()
            try:
                res = minimize(_surrogate_obj, x0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                pts = res.x.reshape(-1, 2)
                _update_best(pts)
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish on exact objective
        if time.time() - t0 < DEADLINE - 10:
            try:
                res = minimize(_score_neg, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                pts = res.x.reshape(-1, 2)
                _update_best(pts)
            except Exception:
                pass
    
    # Phase 4: Multi-start with random perturbations of best
    if best_points is not None and time.time() - t0 < DEADLINE - 20:
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() - t0 > DEADLINE - 15:
                break
            perturbed = best_points + rng.randn(13, 2) * 0.1
            try:
                res = minimize(_score_neg, perturbed.flatten(), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
                pts = res.x.reshape(-1, 2)
                _update_best(pts)
            except Exception:
                pass
    
    # Phase 5: Differential evolution if time permits
    if time.time() - t0 < DEADLINE - 50:
        try:
            bounds = [(-10, 10)] * (2 * 13)
            res = differential_evolution(_score_neg, bounds, seed=42, maxiter=50,
                                        popsize=15, tol=1e-8, polish=False,
                                        updating='deferred', workers=1)
            pts = res.x.reshape(-1, 2)
            _update_best(pts)
        except Exception:
            pass
    
    # Final check
    if best_points is None:
        # Fallback: regular 13-gon
        angles = 2 * np.pi * np.arange(13) / 13
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
