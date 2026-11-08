"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # Safety margin before 150s kill


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


def _coordinate_descent_polish(pts, max_iter=200, tol=1e-10):
    """Coordinate descent to directly maximize min triangle area."""
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iter):
        improved = False
        for idx in range(n):
            best_pos = pts[idx].copy()
            best_score = _score(pts)
            
            # Try small perturbations in each direction
            for dx in [-0.01, 0.01]:
                for dy in [-0.01, 0.01]:
                    trial = pts.copy()
                    trial[idx] = best_pos + np.array([dx, dy])
                    s = _score(trial)
                    if s > best_score + tol:
                        best_score = s
                        best_pos = trial[idx].copy()
                        improved = True
            pts[idx] = best_pos
        
        if not improved:
            break
    
    return pts


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


def _de_objective(pts_flat):
    """Objective for differential evolution: minimize negative score."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    # Add penalty if points are too close together
    from scipy.spatial.distance import pdist
    dmin = pdist(points)
    if np.min(dmin) < 0.01:
        return 1e10 + (0.01 - np.min(dmin)) * 1000
    s = _score(points)
    return -s


def _multi_beta_soft_obj(pts_flat, betas=(50.0, 150.0, 300.0)):
    """Multi-beta soft objective: weighted combination of soft-mins."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    total = 0.0
    for beta in betas:
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min_val = -log_sum / beta
        if soft_min_val < 1e-12:
            return 1e10
        total += hull_area / soft_min_val
    return total / len(betas)


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
    # Create a hexagonal arrangement
    # 14 points: center + ring of 6 + ring of 7 (approx)
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # First ring (6 points)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    # Second ring (7 points to make 14 total)
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _initial_cosine_mod(n):
    """Regular polygon with cosine-modulated radii."""
    angles = 2 * np.pi * np.arange(n) / n
    # Modulate radii with a cosine wave
    mod = 1.0 + 0.1 * np.cos(3 * angles)
    pts = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    return _normalize_unit_area(pts)


def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Differential evolution for global exploration
    if time.time() - start_time < DEADLINE - 30:
        try:
            # Bounds: points in [-2, 2] x [-2, 2]
            bounds = [(-2, 2)] * (2 * N)
            
            de_res = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                update_interval=1,
                callback=None
            )
            
            if de_res.x is not None:
                cand = de_res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starting points
    inits = []
    
    # Use DE result if available
    if best_pts is not None:
        inits.append(best_pts.copy())
    
    # Add various initial configurations
    inits.append(_initial_regular_polygon(N))
    inits.append(_initial_mixed(N))
    inits.append(_initial_hexagonal(N))
    inits.append(_initial_cosine_mod(N))
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.1, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        x0 = init.flatten()
        
        # Nelder-Mead on soft objective
        try:
            res = minimize(_soft_obj, x0, args=(200.0,), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
        
        # Nelder-Mead on hard objective
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            polished = _coordinate_descent_polish(best_pts, max_iter=50, tol=1e-12)
            s = _score(polished)
            if s > best_score:
                best_score = s
                best_pts = polished
        except Exception:
            pass
    
    # Phase 4: Final tight Nelder-Mead
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
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
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
