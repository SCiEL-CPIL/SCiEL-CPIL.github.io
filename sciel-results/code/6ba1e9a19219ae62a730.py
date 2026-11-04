"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))

def _triangle_areas(pts):
    """Compute all triangle areas efficiently."""
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    v1 = pts[j] - pts[i]
    v2 = pts[k] - pts[i]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """Compute the minimum triangle area divided by the convex hull area."""
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

# EVOLVE-BLOCK-START

def _bary_to_pts(bary):
    """
    Convert barycentric coordinates to Cartesian points.
    bary has shape (11, 3): for points 2..12 (0-indexed), bary[i] = (a, b, c)
    with a + b + c = 1, a, b, c > 0.
    Points 0 and 1 are the reference triangle vertices.
    We use a reference triangle with vertices:
      P0 = (0, 0), P1 = (1, 0), P2 = (0.5, h)
    where h is chosen so the reference triangle has area 1.
    Area = 0.5 * base * height = 0.5 * 1 * h = 1 => h = 2.
    So P2 = (0.5, 2.0).
    
    All 13 points lie inside or on the boundary of this reference triangle
    when barycentric coords are all >= 0.
    """
    # Reference triangle vertices
    ref = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, 2.0]])
    # bary shape: (11, 3)
    # Normalize to ensure they sum to 1 (soft constraint)
    b = bary.copy()
    b_sum = b.sum(axis=1, keepdims=True)
    b = b / b_sum
    # Clip to avoid negative values
    b = np.maximum(b, 0.0)
    # Re-normalize after clipping
    b = b / b.sum(axis=1, keepdims=True)
    
    # Points 0 and 1 are fixed
    pts = np.empty((N, 2))
    pts[0] = ref[0]
    pts[1] = ref[1]
    # Points 2..12 from barycentric
    for i in range(11):
        pts[i + 2] = b[i, 0] * ref[0] + b[i, 1] * ref[1] + b[i, 2] * ref[2]
    return pts

def _smooth_min_val(areas, hull_area, tau):
    """Compute smooth minimum of area/hull_area ratios."""
    ratios = areas / hull_area
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -50, 0)
    max_s = np.max(shifted)
    log_sum = max_s + np.log(np.sum(np.exp(shifted - max_s)))
    return min_r - (1.0 / tau) * log_sum

def _objective_bary(bary, tau):
    """
    Objective: minimize -smooth_min(ratios) using barycentric params.
    bary: shape (33,) = 11 * 3 barycentric coords.
    """
    pts = _bary_to_pts(bary)
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-9:
        return 1e9
    areas = _triangle_areas(pts)
    val = _smooth_min_val(areas, hull_area, tau)
    return -val

def _objective_bary_grad(bary, tau):
    """
    Objective with analytic gradient via finite-difference-free approach.
    For L-BFGS-B we provide gradient.
    """
    pts = _bary_to_pts(bary)
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-9:
        return 1e9, np.zeros_like(bary)
    
    areas = _triangle_areas(pts)
    ratios = areas / hull_area
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -50, 0)
    max_s = np.max(shifted)
    log_sum = max_s + np.log(np.sum(np.exp(shifted - max_s)))
    val = min_r - (1.0 / tau) * log_sum
    obj = -val
    
    # Gradient via chain rule:
    # d(obj)/d(bary_k) = -d(smooth_min)/d(bary_k)
    # smooth_min = min_r - (1/tau) * log_sum
    # d(smooth_min)/d(ratio_i) = - (1/tau) * exp(shifted_i) / sum(exp(shifted))
    #                           (for the non-min terms; the min term gives -1 * indicator)
    # Actually the smooth min gradient is:
    # d/dx_i [-1/tau * log(sum exp(-tau x_j))] = exp(-tau x_i) / sum(exp(-tau x_j))
    # So d(smooth_min)/d(ratio_i) = w_i where w_i = exp(shifted_i) / sum(exp(shifted))
    # Note: sum(w_i) = 1, and w_i is concentrated near the minimum.
    
    exp_shifted = np.exp(shifted - max_s)
    w = exp_shifted / np.sum(exp_shifted)  # weights, sum to 1
    
    # d(ratio_i)/d(bary_k) where ratio_i = area_i / hull_area
    # ratio_i = area_i / hull_area
    # d(ratio_i)/d(bary_k) = (d(area_i)/d(bary_k) * hull_area - area_i * d(hull_area)/d(bary_k)) / hull_area^2
    # = (d(area_i)/d(bary_k)) / hull_area - ratio_i * d(hull_area)/d(bary_k) / hull_area
    
    # This is complex. Let's use numerical gradient for the barycentric params
    # since the dimension is only 33.
    grad = np.zeros_like(bary)
    eps = 1e-6
    for k in range(len(bary)):
        bary_plus = bary.copy()
        bary_plus[k] += eps
        obj_plus = _objective_bary(bary_plus, tau)
        bary_minus = bary.copy()
        bary_minus[k] -= eps
        obj_minus = _objective_bary(bary_minus, tau)
        grad[k] = (obj_plus - obj_minus) / (2 * eps)
    
    return obj, grad

def _generate_bary_initial(rng, method='regular'):
    """
    Generate initial barycentric coordinates (shape 33 = 11*3).
    """
    if method == 'regular':
        # Regular 13-gon inscribed in the reference triangle
        # Place 13 points roughly evenly
        bary = np.zeros((11, 3))
        angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
        # Map to barycentric: use a circular pattern in barycentric space
        for i in range(11):
            # Center at (1/3, 1/3, 1/3), radius ~0.25
            r = 0.25
            theta = angles[i]
            bary[i, 0] = 1/3 + r * np.cos(theta)
            bary[i, 1] = 1/3 + r * np.sin(theta)
            bary[i, 2] = 1 - bary[i, 0] - bary[i, 1]
        bary = np.maximum(bary, 0.01)
        bary /= bary.sum(axis=1, keepdims=True)
        return bary.flatten()
    
    elif method == 'grid':
        # Grid in barycentric coordinates
        bary = np.zeros((11, 3))
        # 3x3 grid + 2 extra
        coords = [(0.2, 0.2), (0.2, 0.5), (0.2, 0.8),
                  (0.5, 0.2), (0.5, 0.5), (0.5, 0.8),
                  (0.8, 0.2), (0.8, 0.5), (0.8, 0.8),
                  (0.35, 0.35), (0.65, 0.35)]
        for i in range(11):
            a, b = coords[i]
            bary[i, 0] = a
            bary[i, 1] = b
            bary[i, 2] = 1 - a - b
        bary = np.maximum(bary, 0.01)
        bary /= bary.sum(axis=1, keepdims=True)
        return bary.flatten()
    
    elif method == 'random':
        bary = rng.uniform(0.05, 0.95, (11, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        return bary.flatten()
    
    elif method == 'rings':
        # Two rings in barycentric space
        bary = np.zeros((11, 3))
        # Outer ring: 6 points
        outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            r = 0.35
            theta = outer_angles[i]
            bary[i, 0] = 1/3 + r * np.cos(theta)
            bary[i, 1] = 1/3 + r * np.sin(theta)
            bary[i, 2] = 1 - bary[i, 0] - bary[i, 1]
        # Inner ring: 5 points
        inner_angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        for i in range(5):
            r = 0.15
            theta = inner_angles[i]
            bary[6+i, 0] = 1/3 + r * np.cos(theta)
            bary[6+i, 1] = 1/3 + r * np.sin(theta)
            bary[6+i, 2] = 1 - bary[6+i, 0] - bary[6+i, 1]
        bary = np.maximum(bary, 0.01)
        bary /= bary.sum(axis=1, keepdims=True)
        return bary.flatten()
    
    else:
        bary = rng.uniform(0.05, 0.95, (11, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        return bary.flatten()

def _nelder_mead_polish(bary, tau, max_iter, time_limit):
    """
    Nelder-Mead optimization on barycentric parameters.
    """
    start = time.time()
    
    def obj_fn(x):
        return _objective_bary(x, tau)
    
    try:
        res = minimize(
            obj_fn,
            bary,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True,
            }
        )
        return res.x
    except Exception:
        return bary

def _lbfgs_polish(bary, tau, max_iter, time_limit):
    """
    L-BFGS-B optimization with numerical gradient.
    """
    start = time.time()
    
    def obj_grad(x):
        return _objective_bary_grad(x, tau)
    
    try:
        res = minimize(
            obj_grad,
            bary,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'ftol': 1e-14,
                'gtol': 1e-10,
            }
        )
        return res.x
    except Exception:
        return bary

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -np.inf
    
    def _eval_score(bary):
        """Evaluate the actual score for a barycentric configuration."""
        pts = _bary_to_pts(bary)
        return _compute_min_ratio(pts)
    
    def _time_ok(margin=2.0):
        return (time.time() - start_time) < time_limit - margin
    
    # Phase 1: Multi-start with Nelder-Mead at moderate tau
    tau_schedule = [50, 200, 800, 3000, 10000, 30000, 80000]
    
    methods = ['regular', 'grid', 'rings', 'random', 'regular', 'grid', 'rings', 'random']
    
    candidates = []
    for method in methods:
        if not _time_ok(20.0):
            break
        b = _generate_bary_initial(rng, method)
        # Add small perturbation
        b += rng.normal(0, 0.01, b.shape)
        candidates.append(b)
    
    # Polish each candidate through the tau schedule
    for tau in tau_schedule:
        if not _time_ok(5.0):
            break
        
        for i in range(len(candidates)):
            if not _time_ok(3.0):
                break
            
            try:
                # Use Nelder-Mead for robustness
                max_iter = 300 if tau < 1000 else 500
                candidates[i] = _nelder_mead_polish(
                    candidates[i], tau, max_iter, time_limit
                )
                
                score = _eval_score(candidates[i])
                if score > best_score:
                    best_score = score
                    best_bary = candidates[i].copy()
                    
            except Exception:
                pass
    
    # Phase 2: Fine-tune the best solution with L-BFGS-B
    if best_bary is not None and _time_ok(10.0):
        for tau in [50000, 100000, 200000]:
            if not _time_ok(3.0):
                break
            try:
                max_iter = 400
                best_bary = _lbfgs_polish(
                    best_bary, tau, max_iter, time_limit
                )
                score = _eval_score(best_bary)
                if score > best_score:
                    best_score = score
            except Exception:
                pass
    
    # Phase 3: Perturbation-based restarts on the best solution
    if best_bary is not None:
        perturb_scales = [0.002, 0.005, 0.01, 0.003, 0.008, 0.015]
        for scale in perturb_scales:
            if not _time_ok(2.0):
                break
            try:
                perturbed = best_bary + rng.normal(0, scale, best_bary.shape)
                # Ensure barycentric coords stay positive
                perturbed = np.maximum(perturbed, 0.005)
                perturbed /= perturbed.reshape(11, 3).sum(axis=1, keepdims=True)
                perturbed = perturbed.flatten()
                
                for tau in [50000, 100000]:
                    if not _time_ok(1.5):
                        break
                    perturbed = _nelder_mead_polish(
                        perturbed, tau, 200, time_limit
                    )
                    score = _eval_score(perturbed)
                    if score > best_score:
                        best_score = score
                        best_bary = perturbed.copy()
            except Exception:
                pass
    
    # Final ultra-precise polish
    if best_bary is not None and _time_ok(2.0):
        try:
            best_bary = _lbfgs_polish(best_bary, 300000, 600, time_limit)
            score = _eval_score(best_bary)
            if score > best_score:
                best_score = score
        except Exception:
            pass
    
    # Construct final points
    if best_bary is None:
        best_bary = _generate_bary_initial(rng, 'regular')
    
    final_pts = _bary_to_pts(best_bary)
    
    return final_pts

# EVOLVE-BLOCK-END
