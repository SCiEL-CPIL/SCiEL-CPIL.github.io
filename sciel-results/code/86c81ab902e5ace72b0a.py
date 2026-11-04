"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _triangle_indices(n):
    """Precompute all triangle index triples for n points."""
    ii, jj, kk = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ii.append(i)
                jj.append(j)
                kk.append(k)
    return np.array(ii), np.array(jj), np.array(kk)


_TRI_I, _TRI_J, _TRI_K = _triangle_indices(N)
_NUM_TRI = len(_TRI_I)


def _triangle_areas(points):
    """Compute areas of all C(n,3) triangles."""
    p1 = points[_TRI_I]
    p2 = points[_TRI_J]
    p3 = points[_TRI_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _halton_sequence(dim, count):
    """Generate Halton sequence in 2D."""
    result = np.zeros((count, dim))
    for d in range(dim):
        base = d + 2
        f = 1.0
        for i in range(count):
            f /= base
            result[i, d] = f * (i + 1)
            n = i + 1
            while n > 0:
                f = f * base
                n, r = divmod(n, base)
                if r > 0:
                    f = r / base
    return result


def _halton_points_2d(count, scale=1.0):
    """Generate count points using Halton sequence, scaled to unit disk."""
    h = _halton_sequence(2, count)
    # Map [0,1]^2 to disk using polar coordinates
    r = np.sqrt(h[:, 0])
    theta = 2 * np.pi * h[:, 1]
    pts = scale * np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return points


def _soft_min_objective(points, beta=200.0):
    """Smooth minimum of triangle areas divided by hull area.
    
    Uses log-sum-exp trick for numerical stability.
    Returns (value, gradient) where value = -softmin(areas)/hull_area.
    """
    areas = _triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0, np.zeros(2 * N)
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # For numerical stability, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    
    value = -soft_min_val / hull_area
    
    # Gradient with respect to points
    # d(soft_min)/d(area_i) = exp(-beta * area_i) / sum(exp(-beta * area_j))
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    # d(soft_min)/d(point) = sum_i weights_i * d(area_i)/d(point)
    # area_i = 0.5 * |cross|, for non-degenerate triangles in convex position,
    # cross > 0, so d(area)/d(point) = 0.5 * sign(cross) * d(cross)/d(point)
    
    p1 = points[_TRI_I]
    p2 = points[_TRI_J]
    p3 = points[_TRI_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    # d(cross)/d(p1) = (-v2[:,1], v2[:,0])  (since cross = v1_x*v2_y - v1_y*v2_x)
    # d(cross)/d(p2) = (v2[:,1], -v2[:,0])
    # d(cross)/d(p3) = (-v1[:,1], v1[:,0])
    
    grad = np.zeros(2 * N)
    w = 0.5 * weights * signs / hull_area  # d(soft_min/hull)/d(point) contribution
    
    # Contribution to p1
    grad[2*_TRI_I] += w * (-v2[:, 1])
    grad[2*_TRI_I + 1] += w * v2[:, 0]
    
    # Contribution to p2
    grad[2*_TRI_J] += w * v2[:, 1]
    grad[2*_TRI_J + 1] += w * (-v2[:, 0])
    
    # Contribution to p3
    grad[2*_TRI_K] += w * (-v1[:, 1])
    grad[2*_TRI_K + 1] += w * v1[:, 0]
    
    # We want to minimize -soft_min/hull, so gradient of objective = -gradient of soft_min/hull
    # But we also need to account for hull_area in denominator
    # d(-soft_min/hull)/d(x) = -d(soft_min)/d(x)/hull + soft_min * d(hull)/d(x)/hull^2
    # The second term is complex; for simplicity, use first-order approximation
    # Actually, let's just return the gradient of -soft_min/hull_area
    # which is approximately -grad_soft_min / hull_area (ignoring hull gradient)
    
    # The gradient computed above is d(soft_min)/d(point) / hull_area
    # So d(-soft_min/hull)/d(point) ≈ -grad / hull_area... wait, I already divided by hull_area
    # Let me redo: value = -soft_min / hull_area
    # d(value)/d(point) = -d(soft_min)/d(point) / hull_area + soft_min * d(hull_area)/d(point) / hull_area^2
    # For the first term: -grad_soft_min / hull_area
    # grad above = d(soft_min)/d(point) / hull_area (I included /hull_area in w)
    # So first term = -grad
    
    # For the second term, we need d(hull_area)/d(point) which is complex
    # Let's ignore it for now (it's a smaller effect)
    
    return value, -grad


def _objective_and_grad(x, beta=200.0):
    """Objective and gradient for L-BFGS-B.
    
    x: flat array of 2*N coordinates
    """
    points = x.reshape(N, 2)
    val, grad = _soft_min_objective(points, beta)
    return val, grad


def _exact_objective(x):
    """Exact objective for final evaluation."""
    points = x.reshape(N, 2)
    return -_score(points)


def construct():
    start_time = time.time()
    
    # Generate diverse initial candidates
    candidates = []
    
    # Regular 13-gon
    candidates.append(_regular_polygon(N))
    
    # Halton sequence points
    for scale in [0.8, 0.9, 1.0]:
        candidates.append(_halton_points_2d(N, scale))
    
    # Concentric rings
    for _ in range(3):
        rng = np.random.RandomState(_ * 100 + 7)
        # 6 points on outer ring, 7 on inner ring
        outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + rng.uniform(0, 2*np.pi/6, 1)
        inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + rng.uniform(0, 2*np.pi/7, 1)
        outer_pts = np.column_stack([0.95*np.cos(outer_angles), 0.95*np.sin(outer_angles)])
        inner_pts = np.column_stack([0.45*np.cos(inner_angles), 0.45*np.sin(inner_angles)])
        pts = np.vstack([outer_pts, inner_pts])
        candidates.append(pts)
    
    # Random perturbations of regular polygon
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _regular_polygon(N)
        perturbed = base + rng.randn(N, 2) * 0.08
        candidates.append(perturbed)
    
    # Track best
    best_points = None
    best_score = 0.0
    
    for points in candidates:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Local optimization using L-BFGS-B with smooth minimum
    if best_points is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        x_init = best_points.flatten().copy()
        
        # Use progressively higher beta for sharper minimum
        betas = [50, 100, 200, 400, 800]
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            def obj(x, beta=beta):
                val, grad = _objective_and_grad(x, beta)
                return val, grad
            
            try:
                result = minimize(
                    obj,
                    x_init,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
                )
                
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                opt_points = result.x.reshape(N, 2)
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    x_init = result.x.copy()
                    
            except Exception:
                continue
        
        # Final polish with exact objective (no smooth min)
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            pass
        else:
            # Do a few more iterations with very high beta
            for beta in [1500, 3000]:
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                def obj2(x, beta=beta):
                    val, grad = _objective_and_grad(x, beta)
                    return val, grad
                
                try:
                    result = minimize(
                        obj2,
                        x_init,
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 3000}
                    )
                    
                    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                        break
                    
                    opt_points = result.x.reshape(N, 2)
                    opt_score = _score(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        x_init = result.x.copy()
                        
                except Exception:
                    continue
    
    # Nelder-Mead polish on the best point found
    if best_points is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        x_init = best_points.flatten().copy()
        
        def exact_obj(x):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                return 0.0
            return _exact_objective(x)
        
        try:
            result = minimize(
                exact_obj,
                x_init,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                pass
            else:
                opt_points = result.x.reshape(N, 2)
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
        except Exception:
            pass
    
    # Final fallback
    if best_points is None:
        best_points = _regular_polygon(N)
    
    return best_points
# EVOLVE-BLOCK-END
