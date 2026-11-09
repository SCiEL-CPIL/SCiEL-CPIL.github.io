import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
import itertools
from scipy.optimize import minimize

_DEADLINE = time.time() + 585.0  # Safety margin
_COMBOS = np.array(list(itertools.combinations(range(13), 3)))
_N_COMBOS = len(_COMBOS)

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

def _soft_min_grad(params, temperature):
    """Gradient of the soft-min surrogate objective."""
    if time.time() > _DEADLINE:
        return np.ones_like(params)
    points = params.reshape((13, 2))
    if not np.all(np.isfinite(points)):
        return np.ones_like(params)
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return np.ones_like(params)
    
    idx = _COMBOS
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    
    # Raw (unabsorbed) signed areas for gradient computation
    raw_areas = (a[:, 0] * (b[:, 1] - c[:, 1])
                 + b[:, 0] * (c[:, 1] - a[:, 1])
                 + c[:, 0] * (a[:, 1] - b[:, 1]))
    areas = 0.5 * np.abs(raw_areas)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    exp_vals = np.exp(-shifted / temperature)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # Gradient of soft_min w.r.t. areas: -weights
    # Gradient of areas w.r.t. points:
    # d(area_k)/d(p_i) for each triangle k and point i
    
    grad_points = np.zeros_like(points)
    
    for k in range(_N_COMBOS):
        i1, i2, i3 = idx[k]
        w = -weights[k]  # d(soft_min)/d(area_k)
        
        # d(area_k)/d(p_i) for each vertex
        # area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # For gradient, use sign of raw area
        sign = np.sign(raw_areas[k])
        if sign == 0:
            sign = 1.0
        
        # d(area)/d(x1) = 0.5 * sign * (y2 - y3)
        # d(area)/d(y1) = 0.5 * sign * (x3 - x2)
        # d(area)/d(x2) = 0.5 * sign * (y3 - y1)
        # d(area)/d(y2) = 0.5 * sign * (x1 - x3)
        # d(area)/d(x3) = 0.5 * sign * (y1 - y2)
        # d(area)/d(y3) = 0.5 * sign * (x2 - x1)
        
        grad_points[i1, 0] += 0.5 * sign * (points[i2, 1] - points[i3, 1]) * w
        grad_points[i1, 1] += 0.5 * sign * (points[i3, 0] - points[i2, 0]) * w
        grad_points[i2, 0] += 0.5 * sign * (points[i3, 1] - points[i1, 1]) * w
        grad_points[i2, 1] += 0.5 * sign * (points[i1, 0] - points[i3, 0]) * w
        grad_points[i3, 0] += 0.5 * sign * (points[i1, 1] - points[i2, 1]) * w
        grad_points[i3, 1] += 0.5 * sign * (points[i2, 0] - points[i1, 0]) * w
    
    # Chain rule: objective = -soft_min / ha
    # d(objective)/d(points) = -d(soft_min)/d(points) / ha
    grad_points = -grad_points / ha
    
    return grad_points.flatten()

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
    
    # Delaunay-based structured configs
    for _ in range(3):
        base = _concentric_rings(6, 7, 0.42, 1.0, np.pi/7)
        # Add slight radial perturbation to create more uniform Delaunay mesh
        angles = np.arctan2(base[:, 1], base[:, 0])
        radii = np.sqrt(base[:, 0]**2 + base[:, 1]**2)
        radii += rng.normal(0, 0.03, 13)
        base = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(base)
    
    return configs

def _lloyd_relax(points, max_iter=50, tol=1e-8):
    """Lloyd's algorithm: iteratively move each point to its Voronoi centroid.
    This enforces more uniform local density, which helps maximize min triangle area."""
    if time.time() > _DEADLINE - 60:
        return points
    
    points = points.copy()
    n = len(points)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 60:
            break
        
        try:
            vor = Voronoi(points)
        except Exception:
            break
        
        new_points = points.copy()
        max_shift = 0.0
        
        for i in range(n):
            # Find Voronoi region vertices
            region_idx = vor.regions[vor.point_region[i]]
            if len(region_idx) == 0 or -1 in region_idx:
                continue
            
            region_verts = vor.vertices[region_idx]
            
            # Compute centroid of Voronoi cell
            # Use area-weighted centroid for better results
            if len(region_verts) >= 3:
                # Compute polygon area and centroid
                x = region_verts[:, 0]
                y = region_verts[:, 1]
                x2 = np.roll(x, -1)
                y2 = np.roll(y, -1)
                cross = x * y2 - x2 * y
                area = 0.5 * np.sum(cross)
                
                if abs(area) > 1e-15:
                    cx = np.sum((x + x2) * cross) / (6 * area)
                    cy = np.sum((y + y2) * cross) / (6 * area)
                else:
                    cx = np.mean(x)
                    cy = np.mean(y)
            else:
                cx = np.mean(region_verts[:, 0])
                cy = np.mean(region_verts[:, 1])
            
            # Clip to convex hull to keep points inside
            # Simple check: keep within bounding box of original points
            cx = np.clip(cx, points[:, 0].min() - 0.1, points[:, 0].max() + 0.1)
            cy = np.clip(cy, points[:, 1].min() - 0.1, points[:, 1].max() + 0.1)
            
            new_points[i] = [cx, cy]
            max_shift = max(max_shift, np.linalg.norm(new_points[i] - points[i]))
        
        points = new_points
        
        if max_shift < tol:
            break
    
    return points


def _delaunay_optimize(points, max_iter=30):
    """Optimize using L-BFGS-B with gradient-based soft-min surrogate."""
    if time.time() > _DEADLINE - 45:
        return points
    
    points = points.copy()
    
    # Phase A: Lloyd relaxation for uniform density
    for _ in range(max(3, max_iter // 4)):
        if time.time() > _DEADLINE - 45:
            break
        points = _lloyd_relax(points, max_iter=8)
    
    # Phase B: L-BFGS-B with soft-min surrogate (coarse to fine)
    params = points.flatten()
    for temp in [0.2, 0.08, 0.02]:
        if time.time() > _DEADLINE - 45:
            break
        try:
            res = minimize(lambda p: _soft_min_ratio(p, temp),
                         lambda p: _soft_min_grad(p, temp),
                         params,
                         method='L-BFGS-B',
                         options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
            params = res.x
        except Exception:
            break
    
    # Phase C: Exact refinement with Powell
    try:
        res = minimize(_exact_neg_ratio, params,
                      method='Powell',
                      options={'maxiter': 40, 'xtol': 1e-11, 'ftol': 1e-11})
        opt_points = res.x.reshape((13, 2))
    except Exception:
        opt_points = params.reshape((13, 2))
    
    return opt_points


def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Phase 1: Generate structured configs and evaluate
    configs = _generate_structured_configs(rng)
    
    for config in configs:
        if time.time() > _DEADLINE - 200:
            break
        
        init_score = _ratio(config)
        if init_score > best_score:
            best_score = init_score
            best_points = config.copy()
    
    # Phase 2: L-BFGS-B optimization from best config
    if best_points is not None:
        if time.time() <= _DEADLINE - 120:
            try:
                delaunay_opt = _delaunay_optimize(best_points, max_iter=50)
                delaunay_score = _ratio(delaunay_opt)
                if delaunay_score > best_score:
                    best_score = delaunay_score
                    best_points = delaunay_opt.copy()
            except Exception:
                pass
        
        # Coarse-to-fine L-BFGS-B optimization
        if time.time() <= _DEADLINE - 60:
            params = best_points.flatten()
            
            for temp in [0.3, 0.1, 0.03, 0.01]:
                if time.time() > _DEADLINE - 60:
                    break
                
                try:
                    res = minimize(lambda p: _soft_min_ratio(p, temp),
                                 lambda p: _soft_min_grad(p, temp),
                                 params,
                                 method='L-BFGS-B',
                                 options={'maxiter': 120, 'gtol': 1e-9, 'ftol': 1e-11})
                    params = res.x
                except Exception:
                    continue
            
            # Fine phase: exact objective with Powell
            try:
                res = minimize(_exact_neg_ratio, params,
                              method='Powell',
                              options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12})
                opt_points = res.x.reshape((n, 2))
                opt_score = _ratio(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
            except Exception:
                pass
    
    # Phase 3: Multi-start with L-BFGS-B optimization from different configs
    for config in configs[:8]:
        if time.time() > _DEADLINE - 30:
            break
        
        try:
            opt_points = _delaunay_optimize(config, max_iter=30)
            opt_score = _ratio(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            continue
    
    # Fallback
    if best_points is None or best_score < 0.001:
        best_points = _regular_polygon(13, radius=1.0)
        best_score = _ratio(best_points)
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=1.0)
        
    return best_points
