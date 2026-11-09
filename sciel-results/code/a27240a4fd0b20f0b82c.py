import time
import numpy as np
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_DEADLINE = time.time() + 585.0  # Safety margin
_COMBOS = np.array(list(itertools.combinations(range(13), 3)))

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = _COMBOS
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _ratio(points):
    """Compute min_triangle_area / hull_area. Returns 0 if invalid."""
    if not np.all(np.isfinite(points)):
        return 0.0
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    if ma < 1e-12:
        return 0.0
    return ma / ha

def _soft_min_ratio(params, temperature):
    """Smooth surrogate: soft-min of triangle areas / hull area.
    Returns negative value to minimize."""
    if time.time() > _DEADLINE:
        return 1e10
    points = params.reshape((13, 2))
    if not np.all(np.isfinite(points)):
        return 1e10
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10
    
    idx = _COMBOS
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    
    # Soft min: -log(sum(exp(-areas/T)))/T approximates min as T->0
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_a - temperature * log_sum
    
    if soft_min < 1e-12:
        return 1e10
    
    return -(soft_min / ha)

def _exact_neg_ratio(params):
    """Exact objective for final refinement."""
    if time.time() > _DEADLINE:
        return 1e10
    points = params.reshape((13, 2))
    r = _ratio(points)
    if r < 1e-12:
        return 1e10
    return -r

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _concentric_rings(n_inner, n_outer, r_inner=0.3, r_outer=1.0, angle_offset=0.0):
    """Points on two concentric circles with optional angle offset for inner ring."""
    inner_angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    outer = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    return np.vstack([inner, outer])

def _generate_structured_configs(rng):
    """Generate diverse structured initial configurations."""
    configs = []
    
    # Regular 13-gon
    configs.append(_regular_polygon(13, radius=1.0))
    
    # Concentric rings with various ratios and offsets
    for n_in, n_out, r_in, offset in [
        (6, 7, 0.35, 0.0),
        (6, 7, 0.40, np.pi/7),
        (5, 8, 0.35, 0.0),
        (5, 8, 0.40, np.pi/8),
        (4, 9, 0.30, 0.0),
        (4, 9, 0.35, np.pi/9),
        (7, 6, 0.30, 0.0),
        (7, 6, 0.35, np.pi/6),
        (3, 10, 0.25, 0.0),
        (3, 10, 0.30, np.pi/10),
    ]:
        configs.append(_concentric_rings(n_in, n_out, r_in, 1.0, offset))
    
    # Perturbed versions
    for _ in range(5):
        base = _concentric_rings(6, 7, 0.38, 1.0, np.pi/7)
        perturbed = base + rng.normal(0, 0.05, (13, 2))
        configs.append(perturbed)
    
    return configs

def _lbfgs_optimize(params, temperature, maxiter=500, tol=1e-12):
    """Run L-BFGS-B optimization on the soft-min surrogate."""
    if time.time() > _DEADLINE - 60:
        return params
    
    try:
        res = minimize(lambda p: _soft_min_ratio(p, temperature), params,
                      method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol})
        return res.x
    except Exception:
        return params

def _polish_points(points, maxiter=200):
    """High-precision coordinate polish using Powell on exact objective."""
    if time.time() > _DEADLINE - 30:
        return points
    
    params = points.flatten()
    try:
        res = minimize(_exact_neg_ratio, params,
                      method='Powell',
                      options={'maxiter': maxiter, 'xtol': 1e-12, 'ftol': 1e-12})
        opt_points = res.x.reshape((13, 2))
        if np.all(np.isfinite(opt_points)):
            return opt_points
    except Exception:
        pass
    return points

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Phase 1: Generate structured configs and evaluate
    configs = _generate_structured_configs(rng)
    
    for config in configs:
        if time.time() > _DEADLINE - 120:
            break
        
        init_score = _ratio(config)
        if init_score > best_score:
            best_score = init_score
            best_points = config.copy()
    
    # Phase 2: Coarse-to-fine L-BFGS-B optimization using soft-min surrogate
    if best_points is not None:
        params = best_points.flatten()
        
        # Coarse phase: larger temperature for smoother landscape
        for temp in [0.5, 0.1, 0.01, 0.001]:
            if time.time() > _DEADLINE - 90:
                break
            
            params = _lbfgs_optimize(params, temperature=temp, maxiter=300, tol=1e-10)
        
        # Fine phase: very low temperature for near-exact soft-min
        if time.time() > _DEADLINE - 60:
            pass
        else:
            params = _lbfgs_optimize(params, temperature=0.0001, maxiter=500, tol=1e-12)
        
        # High-precision polish with exact objective
        opt_points = _polish_points(params.reshape((n, 2)), maxiter=100)
        opt_score = _ratio(opt_points)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # Phase 3: Multi-start with L-BFGS-B from different structured configs
    for config in configs[:10]:
        if time.time() > _DEADLINE - 45:
            break
        
        params = config.flatten()
        
        # Quick L-BFGS-B optimization with moderate temperature
        params = _lbfgs_optimize(params, temperature=0.05, maxiter=200, tol=1e-10)
        
        # Polish
        opt_points = _polish_points(params.reshape((n, 2)), maxiter=50)
        opt_score = _ratio(opt_points)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # Fallback
    if best_points is None or best_score < 0.001:
        best_points = _regular_polygon(13, radius=1.0)
        best_score = _ratio(best_points)
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=1.0)
        
    return best_points
