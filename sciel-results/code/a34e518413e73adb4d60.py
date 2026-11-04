import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute all triangle index combinations
_TRI_IDX = np.array(list(combinations(range(N), 3)))
_NUM_TRIS = len(_TRI_IDX)

def _all_triangle_areas(pts):
    """Compute areas of all C(N,3) triangles as a 1D array (vectorized)."""
    p1 = pts[_TRI_IDX[:, 0]]
    p2 = pts[_TRI_IDX[:, 1]]
    p3 = pts[_TRI_IDX[:, 2]]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    return float(np.min(_all_triangle_areas(pts)))

def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    """Calculate the objective score: min_triangle_area / hull_area."""
    min_area = _min_triangle_area(pts)
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return min_area / h_area

def _hull_area_grad(pts):
    """
    Compute gradient of convex hull area w.r.t. points.
    Only hull vertices have non-zero gradient.
    For a convex polygon with vertices v_0, ..., v_{m-1} in CCW order:
    A = 0.5 * sum_i (v_i x v_{i+1})
    dA/dv_i = 0.5 * (v_{i+1} - v_{i-1})
    """
    try:
        hull = ConvexHull(pts)
        hull_idx = hull.vertices
        m = len(hull_idx)
        if m < 3:
            return np.zeros_like(pts)
        # Get hull vertices in order
        hull_pts = pts[hull_idx]
        # Compute gradient for each hull vertex
        grad = np.zeros_like(pts)
        for i in range(m):
            prev_idx = hull_idx[(i - 1) % m]
            next_idx = hull_idx[(i + 1) % m]
            # dA/dv_i = 0.5 * (v_{i+1} - v_{i-1})
            grad[hull_idx[i]] = 0.5 * (pts[next_idx] - pts[prev_idx])
        return grad
    except Exception:
        return np.zeros_like(pts)

def _soft_min_score(pts, beta=50.0):
    """
    Smooth approximation of min_triangle_area / hull_area using log-sum-exp.
    soft_min(x) = -(1/beta) * log(sum(exp(-beta * x_i)))
    This is differentiable and approximates min as beta -> inf.
    """
    areas = _all_triangle_areas(pts)
    # Avoid log(0) issues
    areas = np.maximum(areas, 1e-15)
    # Soft minimum: -(1/beta) * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0
    return soft_min_area / h_area

def _neg_soft_min_score(x, beta=50.0):
    """Negative soft-min score for minimization."""
    pts = x.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _neg_soft_min_score_grad(x, beta=50.0):
    """
    Negative soft-min score with analytical gradient.
    f = soft_min_area / hull_area
    df/dx = (grad_soft_min * hull_area - soft_min_area * grad_hull) / hull_area^2
    """
    pts = x.reshape(N, 2)
    areas = _all_triangle_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # Soft min gradient: d(soft_min)/d(areas_i) = exp(-beta*areas_i) / sum(exp(-beta*areas_j))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_vals = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # d(soft_min)/d(areas_i) = -weights_i (since soft_min = -log_sum_exp/beta)
    # Actually: soft_min = -(1/beta)*log(sum(exp(-beta*a)))
    # d(soft_min)/da_i = -(1/beta) * (-beta * exp(-beta*a_i)) / sum(exp(-beta*a)) = exp(-beta*a_i)/sum(exp(-beta*a))
    # So d(soft_min)/da_i = weights_i
    
    # Gradient of areas w.r.t. points
    # area_i = 0.5 * |cross(p2-p1, p3-p1)| where p1=pts[idx_i,0], p2=pts[idx_i,1], p3=pts[idx_i,2]
    # For each triangle, we need d(area)/d(pts)
    # cross = (p2x-p1x)*(p3y-p1y) - (p2y-p1y)*(p3x-p1x)
    # area = 0.5*|cross|
    # d(area)/d(p1) = -0.5*sign(cross) * d(cross)/d(p1)
    # d(cross)/d(p1x) = -(p3y-p1y), d(cross)/d(p1y) = (p3x-p1x)
    # d(cross)/d(p2x) = (p3y-p1y), d(cross)/d(p2y) = -(p3x-p1x)
    # d(cross)/d(p3x) = -(p2y-p1y), d(cross)/d(p3y) = (p2x-p1x)
    
    p1 = pts[_TRI_IDX[:, 0]]
    p2 = pts[_TRI_IDX[:, 1]]
    p3 = pts[_TRI_IDX[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0
    
    grad_areas = np.zeros_like(pts)
    # d(area)/d(pts) for each triangle
    # For point p1: d(area)/dp1x = -0.5*sign*(p3y-p1y), d(area)/dp1y = 0.5*sign*(p3x-p1x)
    # For point p2: d(area)/dp2x = 0.5*sign*(p3y-p1y), d(area)/dp2y = -0.5*sign*(p3x-p1x)
    # For point p3: d(area)/dp3x = -0.5*sign*(p2y-p1y), d(area)/dp3y = 0.5*sign*(p2x-p1x)
    
    w = weights * 0.5 * sign_cross  # combined weight
    
    # Accumulate gradients for p1
    np.add.at(grad_areas, _TRI_IDX[:, 0], -w[:, None] * np.stack([(p3[:, 1] - p1[:, 1]), (p3[:, 0] - p1[:, 0])], axis=1))
    # Accumulate gradients for p2
    np.add.at(grad_areas, _TRI_IDX[:, 1], w[:, None] * np.stack([(p3[:, 1] - p1[:, 1]), -(p3[:, 0] - p1[:, 0])], axis=1))
    # Accumulate gradients for p3
    np.add.at(grad_areas, _TRI_IDX[:, 2], -w[:, None] * np.stack([(p2[:, 1] - p1[:, 1]), (p2[:, 0] - p1[:, 0])], axis=1))
    
    # grad_soft_min_area = grad_areas (since d(soft_min)/d(areas_i) = weights_i, and we already multiplied)
    # Wait, let me redo: d(soft_min)/d(pts) = sum_i d(soft_min)/d(areas_i) * d(areas_i)/d(pts)
    # = sum_i weights_i * d(areas_i)/d(pts)
    # grad_areas above already has the 0.5*sign factor, so it's d(areas_i)/d(pts) summed with weights
    # Actually grad_areas = sum_i weights_i * 0.5 * sign_i * (geometric terms) = sum_i d(areas_i)/d(pts) * weights_i
    # So grad_soft_min = grad_areas
    
    soft_min_area = float(-np.log(sum_exp) / beta - max_val / beta)
    
    h_area = _hull_area(pts)
    if h_area < 1e-10:
        return 0.0, np.zeros_like(pts)
    
    # f = soft_min_area / h_area
    # df/d(pts) = (grad_soft_min * h_area - soft_min_area * grad_hull) / h_area^2
    grad_hull = _hull_area_grad(pts)
    grad_f = (grad_areas * h_area - soft_min_area * grad_hull) / (h_area * h_area)
    
    return -soft_min_area / h_area, -grad_f

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _center_ring(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
    outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0,1)."""
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n+1):
        f /= base
        seq[i-1] = 0.0
        j = i
        while j > 0:
            seq[i-1] += (j % base) * f
            j //= base
    return seq

def _halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    best_score = 0.0
    best_points = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(N, 1.0))
    
    # 2. Concentric rings (various splits)
    for n_outer in [7, 8, 9, 10]:
        n_inner = N - n_outer
        if n_inner >= 3:
            candidates.append(_center_ring(n_outer, n_inner, 1.0, 0.5))
            candidates.append(_center_ring(n_outer, n_inner, 1.0, 0.6))
            candidates.append(_center_ring(n_outer, n_inner, 1.0, 0.4))
    
    # 3. Halton sequence points (low-discrepancy)
    halton = _halton_points(N)
    candidates.append(halton)
    # Scaled Halton
    candidates.append(halton * 2)
    
    # 4. Random points on circle (sorted by angle for convex position)
    rng = np.random.default_rng(42)
    for i in range(15):
        angles = np.sort(rng.uniform(0, 2*np.pi, N))
        radii = rng.uniform(0.7, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 5. Random points in disk
    for i in range(10):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Perturbed regular polygon
    for i in range(10):
        noise = rng.normal(0, 0.05, (N, 2))
        candidates.append(_regular_polygon(N, 1.0) + noise)
    
    # Evaluate all candidates with exact score
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        update_best(pts, s)
    
    # Optimize from best candidates using L-BFGS-B with soft-min surrogate
    # Sort candidates by score, optimize top ones
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Use top candidates as starting points for optimization
    top_candidates = [pts for _, pts in scored_candidates[:min(10, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        for init_pts in top_candidates:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    _neg_soft_min_score_grad,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                update_best(res_pts, res_score)
                
            except Exception:
                pass
        
        # Update top candidates with current best
        if best_points is not None:
            top_candidates = [best_points.copy()]
    
    # Final refinement: use exact score with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 15:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000, 'adaptive': True}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            update_best(res_pts, res_score)
        except Exception:
            pass
    
    # Additional local refinement: perturb each point slightly and re-optimize
    if best_points is not None and time.time() < deadline - 5:
        rng2 = np.random.default_rng(123)
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            # Small random perturbation
            perturbed = best_points + rng2.normal(0, 0.01, (N, 2))
            try:
                res = minimize(
                    lambda x: -_score(x.reshape(N, 2)),
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 3000, 'adaptive': True}
                )
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                update_best(res_pts, res_score)
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
        best_score = _score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
