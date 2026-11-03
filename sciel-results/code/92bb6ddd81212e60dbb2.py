"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 142.0  # Safety margin before 150s kill


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


def _soft_obj(pts_flat, beta=100.0):
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


def _local_refine(pts, max_iter=200, tol=1e-12):
    """Efficient local refinement using Nelder-Mead on the hard objective."""
    pts = pts.copy()
    x0 = pts.flatten()
    try:
        res = minimize(_neg_score, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14,
                              'adaptive': True})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand)):
                return cand
    except Exception:
        pass
    return pts


def construct():
    start_time = time.time()
    
    # Use 4 diverse initializations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
        _initial_hexagonal(N),
        _initial_cosine_mod(N),
    ]
    
    # Add 2 random perturbations of the regular polygon
    rng = np.random.RandomState(42)
    for _ in range(2):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.08, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Optimize from each initialization with soft objective
    for init in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        x0 = init.flatten()
        try:
            # Use moderate beta for good gradient signal
            res = minimize(lambda x: _soft_obj(x, beta=80.0), x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                if np.all(np.isfinite(cand)):
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
        except Exception:
            pass
    
    # Phase 2: Refine the best solution with hard objective
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        try:
            refined = _local_refine(best_pts, max_iter=5000, tol=1e-13)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_pts = refined
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                if np.all(np.isfinite(cand)):
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
