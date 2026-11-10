import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # Safety margin for 630s limit

def _get_triangles(n=13):
    """Pre-compute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

TRI_INDICES = _get_triangles()

def _compute_exact_score(points):
    """
    Computes the exact score: A_min / A_hull.
    Returns (score, A_min, A_hull, min_indices).
    """
    n = len(points)
    if n < 3:
        return 0.0, 0.0, 0.0, (0, 1, 2)
    
    a = points[TRI_INDICES[:, 0]]
    b = points[TRI_INDICES[:, 1]]
    c = points[TRI_INDICES[:, 2]]
    
    # Vectorized area calculation
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    
    A_min = np.min(areas)
    min_idx = np.argmin(areas)
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0, 0.0, 0.0, (0, 1, 2)
        
    if A_hull <= 1e-12:
        return 0.0, 0.0, 0.0, (0, 1, 2)
        
    score = A_min / A_hull
    return score, A_min, A_hull, TRI_INDICES[min_idx]

def _soft_min_objective(params):
    """
    Objective for the global phase.
    Minimizes -soft_min(A_min) / A_hull.
    Uses log-sum-exp for numerical stability.
    """
    points = params.reshape(13, 2)
    
    if not np.all(np.isfinite(points)):
        return 1e10
        
    a = points[TRI_INDICES[:, 0]]
    b = points[TRI_INDICES[:, 1]]
    c = points[TRI_INDICES[:, 2]]
    
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    
    # Soft min with temperature T
    T = 0.05
    neg_areas_over_T = -areas / T
    M = np.max(neg_areas_over_T)
    if M == -np.inf:
        return 1e10
    log_sum = M + np.log(np.sum(np.exp(neg_areas_over_T - M)))
    soft_min_area = -T * log_sum
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
        
    if A_hull <= 1e-9:
        return 1e10
        
    ratio = soft_min_area / A_hull
    return -ratio

def _exact_objective(params):
    """
    Objective for the local polishing phase.
    Minimizes -A_min / A_hull.
    """
    points = params.reshape(13, 2)
    score, _, _, _ = _compute_exact_score(points)
    return -score

def _exact_gradient(params):
    """
    Computes the subgradient of -A_min / A_hull.
    
    Let f = A_min / A_hull.
    We want grad(-f) = -grad(f).
    
    If the minimum triangle is unique (or we pick one), A_min is locally
    the area of that specific triangle.
    grad(A_min) = grad(A_{i,j,k})
    grad(A_hull) is approximated numerically because analytic hull gradients
    are complex and piecewise. However, since we are in a polishing phase
    where the hull is likely stable, numerical gradient of A_hull is sufficient
    and robust.
    
    Actually, for speed and robustness, let's just use numerical gradient
    for the exact objective in the polishing phase? 
    No, the prompt asks for a structural change. 
    Using analytic gradient for the numerator and numerical for denominator
    is a good hybrid.
    
    Let's compute analytic gradient for A_min (the min triangle).
    And numerical gradient for A_hull.
    
    grad(f) = (grad(A_min) * A_hull - A_min * grad(A_hull)) / A_hull^2
    """
    points = params.reshape(13, 2)
    score, A_min, A_hull, min_tri = _compute_exact_score(points)
    
    if score == 0.0:
        return np.zeros(26)
        
    # Gradient of A_min w.r.t points
    # A = 0.5 * |det(b-a, c-a)|
    # Let v1 = b-a, v2 = c-a.
    # det = v1x*v2y - v1y*v2x
    # grad(A) w.r.t a, b, c.
    
    grad_A_min = np.zeros((13, 2))
    
    i, j, k = min_tri
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    det = v1[0]*v2[1] - v1[1]*v2[0]
    
    # Area = 0.5 * |det|
    # d(Area)/d(p_i) = 0.5 * sign(det) * (-v2[1], v2[0]) ?
    # Let's derive:
    # Area = 0.5 * |x_j(y_k - y_i) + x_k(y_i - y_j) + x_i(y_j - y_k)|
    # Let's use the cross product form.
    # A = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # If det > 0:
    # dA/dp_i = 0.5 * (-(p_k - p_i)_y, (p_k - p_i)_x) ... wait.
    # Let's use standard result for triangle area gradient.
    # grad_p_i (0.5 * det(p_j-p_i, p_k-p_i))
    # = 0.5 * ( - (p_k-p_i)_y, (p_k-p_i)_x )  -- if det > 0?
    # Let's check.
    # f = 0.5 * ( (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) )
    # df/dx_i = 0.5 * ( -(y_k-y_i) - (y_j-y_i)*(-1) ) = 0.5 * ( -y_k + y_i + y_j - y_i ) = 0.5 * (y_j - y_k)
    # df/dy_i = 0.5 * ( (x_j-x_i) - (x_k-x_i) ) = 0.5 * (x_j - x_k)
    
    # So if det > 0:
    # grad_i = 0.5 * (y_j - y_k, x_j - x_k)
    # grad_j = 0.5 * (y_k - y_i, x_k - x_i)
    # grad_k = 0.5 * (y_i - y_j, x_i - x_j)
    
    # If det < 0, we take absolute value, so gradients flip sign.
    sign = 1.0 if det > 0 else -1.0
    
    grad_A_min[i] = 0.5 * sign * (points[j, 1] - points[k, 1], points[j, 0] - points[k, 0])
    grad_A_min[j] = 0.5 * sign * (points[k, 1] - points[i, 1], points[k, 0] - points[i, 0])
    grad_A_min[k] = 0.5 * sign * (points[i, 1] - points[j, 1], points[i, 0] - points[j, 0])
    
    # Numerical gradient for A_hull
    # Central difference
    eps = 1e-6
    grad_A_hull = np.zeros(26)
    
    # To save time, we can compute A_hull once per perturbation.
    # 26 * 2 = 52 evaluations. ConvexHull is fast.
    for idx in range(26):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[idx] += eps
        p_minus[idx] -= eps
        
        try:
            h_plus = ConvexHull(p_plus.reshape(13, 2)).volume
            h_minus = ConvexHull(p_minus.reshape(13, 2)).volume
            grad_A_hull[idx] = (h_plus - h_minus) / (2 * eps)
        except Exception:
            # If hull fails, gradient is undefined or 0? 
            # If hull fails, the point is invalid. 
            # In polishing, we assume validity.
            grad_A_hull[idx] = 0.0
            
    # grad(f) = (grad_A_min * A_hull - A_min * grad_A_hull) / A_hull^2
    grad_f = (grad_A_min.flatten() * A_hull - A_min * grad_A_hull) / (A_hull * A_hull)
    
    # We are minimizing -f
    return -grad_f

def heilbronn_convex13() -> np.ndarray:
    best_points = None
    best_score = -np.inf
    
    # 1. Generate diverse initial points
    # The space is 26D. We need good starts.
    # Start 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    p0_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Start 2: Two concentric rings (6+6+1)
    # Center + Ring 1 (6 pts) + Ring 2 (6 pts)
    p0_rings = np.zeros((13, 2))
    p0_rings[0] = [0, 0]
    for i in range(6):
        th = i * np.pi / 3
        p0_rings[i+1] = [0.5 * np.cos(th), 0.5 * np.sin(th)]
    for i in range(6):
        th = i * np.pi / 3 + np.pi/6
        p0_rings[i+7] = [1.0 * np.cos(th), 1.0 * np.sin(th)]
    
    # Start 3: Random jittered regular polygon
    rng = np.random.default_rng(123)
    p0_jitter = p0_regular + rng.normal(0, 0.1, (13, 2))
    
    # Start 4: Another random configuration
    p0_rand = rng.uniform(-1, 1, (13, 2))
    
    inits = [p0_regular.flatten(), p0_rings.flatten(), p0_jitter.flatten(), p0_rand.flatten()]
    
    # Add a few more random inits
    for _ in range(6):
        p0_extra = p0_regular + rng.normal(0, 0.2, (13, 2))
        inits.append(p0_extra.flatten())
    
    for init_params in inits:
        if time.time() > _DEADLINE - 20:
            break
            
        # Phase 1: Global optimization with soft-min surrogate
        try:
            res1 = minimize(
                _soft_min_objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            p1 = res1.x
        except Exception:
            p1 = init_params
            
        # Phase 2: Local polishing with exact objective and analytic/numeric gradient
        # Use BFGS or L-BFGS-B with gradient
        try:
            res2 = minimize(
                _exact_objective,
                p1,
                jac=_exact_gradient,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            p2 = res2.x
        except Exception:
            p2 = p1
            
        # Evaluate final score
        points_final = p2.reshape(13, 2)
        if not np.all(np.isfinite(points_final)):
            continue
            
        score, _, _, _ = _compute_exact_score(points_final)
        
        if score > best_score:
            best_score = score
            best_points = points_final.copy()
            
        # Also check the result of phase 1 just in case phase 2 degraded it
        points_p1 = p1.reshape(13, 2)
        if np.all(np.isfinite(points_p1)):
            score_p1, _, _, _ = _compute_exact_score(points_p1)
            if score_p1 > best_score:
                best_score = score_p1
                best_points = points_p1.copy()

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
