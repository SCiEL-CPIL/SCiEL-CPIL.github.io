"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, L_BFGS_B
from scipy.spatial import ConvexHull
from itertools import combinations

DEADLINE = 135.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _soft_obj_and_grad(pts_flat, beta=150.0):
    """
    Smooth soft-min objective with analytical gradient.
    Objective: hull_area / soft_min(areas)
    where soft_min uses log-sum-exp with parameter beta.
    """
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10, np.zeros_like(pts_flat)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute all triangle areas and their gradients
    n = N
    areas = np.zeros((n * (n-1) * (n-2)) // 6)
    idx = 0
    grad_x = np.zeros_like(x)
    grad_y = np.zeros_like(y)
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross|
                cross = (x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i])
                a = 0.5 * abs(cross)
                areas[idx] = a
                idx += 1
    
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    
    # Soft min via log-sum-exp
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        exps = np.exp(z - max_z)
        log_sum = max_z + np.log(np.sum(exps))
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10, np.zeros_like(pts_flat)
    
    obj = hull_area / soft_min_val
    
    # Compute gradient numerically (central differences) for reliability
    grad = np.zeros_like(pts_flat)
    h = 1e-7
    for d in range(len(pts_flat)):
        pts_plus = pts_flat.copy()
        pts_minus = pts_flat.copy()
        pts_plus[d] += h
        pts_minus[d] -= h
        f_plus = _soft_obj(pts_plus, beta)
        f_minus = _soft_obj(pts_minus, beta)
        grad[d] = (f_plus - f_minus) / (2 * h)
    
    return obj, grad


def _soft_obj(pts_flat, beta=150.0):
    """Soft objective: hull_area / soft_min(triangle areas)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points for better minimum triangle area."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_hexagonal(n):
    """Hexagonal lattice inspired initialization."""
    pts = []
    pts.append([0.0, 0.0])
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _initial_cosine_mod(n):
    """Regular polygon with cosine-modulated radii."""
    angles = 2 * np.pi * np.arange(n) / n
    mod = 1.0 + 0.1 * np.cos(3 * angles)
    pts = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    return _normalize_unit_area(pts)


def _two_ring_parameterize(params):
    """
    Two-ring parameterization: 7 outer points + 7 inner points.
    params: [r_outer, r_inner, phase_outer, phase_inner, ...perturbations]
    This exploits likely rotational symmetry.
    """
    n_outer = 7
    n_inner = 7
    r_outer = params[0]
    r_inner = params[1]
    phase_outer = params[2]
    phase_inner = params[3]
    
    # Small perturbations to break perfect symmetry slightly
    perturb = params[4:]
    
    outer_angles = 2 * np.pi * np.arange(n_outer) / n_outer + phase_outer
    inner_angles = 2 * np.pi * np.arange(n_inner) / n_inner + phase_inner
    
    # Add small radial perturbations from params
    outer_r = r_outer * (1.0 + 0.01 * np.sin(np.arange(n_outer) * 2.0 + perturb[0] if len(perturb) > 0 else 0))
    inner_r = r_inner * (1.0 + 0.01 * np.sin(np.arange(n_inner) * 2.0 + perturb[1] if len(perturb) > 1 else 0))
    
    outer_pts = np.column_stack([outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)])
    inner_pts = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return _normalize_unit_area(pts)


def _soft_obj_two_ring(params, beta=150.0):
    """Soft objective for two-ring parameterization."""
    pts = _two_ring_parameterize(params)
    return _soft_obj(pts.flatten(), beta)


def construct():
    start_time = time.time()
    
    # Try multiple initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
        _initial_hexagonal(N),
        _initial_cosine_mod(N),
    ]
    
    # Also try a few random perturbations of the regular polygon
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.05, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: L-BFGS-B with smooth soft-min objective from each init
    for init in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        x0 = init.flatten()
        try:
            # L-BFGS-B with soft objective (smooth)
            res = minimize(_soft_obj, x0, method='L-BFGS-B', args=(150.0,),
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                  'maxls': 50})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
        
        # Nelder-Mead with multi-beta soft objective
        if time.time() - start_time > DEADLINE - 25:
            break
        try:
            res = minimize(lambda x: _soft_obj(x, beta=200.0), x0, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 2: Two-ring parameterization with L-BFGS-B
    if time.time() - start_time < DEADLINE - 40:
        # Initial guess for two-ring: outer radius ~1, inner radius ~0.5
        two_ring_inits = [
            np.array([1.0, 0.5, 0.0, np.pi/7.0, 0.0, 0.0]),
            np.array([1.0, 0.6, 0.0, np.pi/7.0, 0.1, -0.1]),
            np.array([1.1, 0.4, 0.0, 0.0, 0.05, 0.05]),
        ]
        
        for tr_init in two_ring_inits:
            if time.time() - start_time > DEADLINE - 35:
                break
            try:
                res = minimize(_soft_obj_two_ring, tr_init, method='L-BFGS-B', args=(200.0,),
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                      'maxls': 50})
                if res.x is not None:
                    cand = _two_ring_parameterize(res.x)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass
    
    # Phase 3: High-precision polish with Nelder-Mead on hard objective
    if best_pts is not None and time.time() - start_time < DEADLINE - 20:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 4: Final polish with very high beta soft objective
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        try:
            x0 = best_pts.flatten()
            res = minimize(lambda x: _soft_obj(x, beta=500.0), x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                  'maxls': 50})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 5: Final tight Nelder-Mead polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-13,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
