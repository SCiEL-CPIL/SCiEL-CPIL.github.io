"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 145.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


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


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    areas = _all_triangle_areas(points)
    return float(np.min(areas)) if len(areas) > 0 else float('inf')


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


def _de_objective(pts_flat):
    """Objective for differential evolution: minimize negative score."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    from scipy.spatial.distance import pdist
    dmin = np.min(pdist(points))
    if dmin < 0.005:
        return 1e10 + (0.005 - dmin) * 1000
    s = _score(points)
    return -s


def _epigraph_obj(pts_flat, t):
    """Epigraph formulation: minimize t subject to all triangle areas >= t * hull_area."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    # We want to maximize min_area / hull_area, so minimize t where t >= hull_area/min_area
    # Actually, let's use: minimize (hull_area / min_area) with soft penalty
    ratio = hull_area / max(min_area, 1e-12)
    return ratio


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


def _polish_critical_triangles(pts, max_iter=200, tol=1e-10):
    """Polish by focusing on the critical (smallest) triangles."""
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        current_score = _score(pts)
        areas = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        
        # Find which triangle is critical
        # Reconstruct triangle indices from linear index
        count = 0
        crit_i = crit_j = crit_k = 0
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if count == min_idx:
                        crit_i, crit_j, crit_k = i, j, k
                        break
                    count += 1
                else:
                    continue
                break
            else:
                continue
            break
        
        # Try to increase the critical triangle area by moving its vertices
        # or moving other points away
        improved = False
        for idx in [crit_i, crit_j, crit_k]:
            for dx in [-0.005, 0.005]:
                for dy in [-0.005, 0.005]:
                    trial = pts.copy()
                    trial[idx] = pts[idx] + np.array([dx, dy])
                    s = _score(trial)
                    if s > current_score + tol:
                        current_score = s
                        pts = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try moving non-critical points to increase hull area or avoid creating smaller triangles
            for idx in range(n):
                if idx in [crit_i, crit_j, crit_k]:
                    continue
                for dx in [-0.003, 0.003]:
                    for dy in [-0.003, 0.003]:
                        trial = pts.copy()
                        trial[idx] = pts[idx] + np.array([dx, dy])
                        s = _score(trial)
                        if s > current_score + tol:
                            current_score = s
                            pts = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return pts


def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Differential evolution for global exploration
    if time.time() - start_time < DEADLINE - 40:
        try:
            bounds = [(-2, 2)] * (2 * N)
            
            de_res = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=80,
                popsize=20,
                tol=1e-5,
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
    
    if best_pts is not None:
        inits.append(best_pts.copy())
    
    inits.append(_initial_regular_polygon(N))
    inits.append(_initial_mixed(N))
    inits.append(_initial_hexagonal(N))
    inits.append(_initial_cosine_mod(N))
    
    rng = np.random.RandomState(42)
    for _ in range(8):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.15, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    # Sort inits by initial score, best first
    scored_inits = []
    for init in inits:
        s = _score(init)
        scored_inits.append((s, init))
    scored_inits.sort(key=lambda x: -x[0])
    
    for init_score, init in scored_inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        x0 = init.flatten()
        
        # Nelder-Mead on soft objective with moderate beta
        try:
            res = minimize(_soft_obj, x0, args=(80.0,), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11,
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
        if time.time() - start_time > DEADLINE - 30:
            break
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 3: Critical triangle polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        try:
            polished = _polish_critical_triangles(best_pts, max_iter=100, tol=1e-11)
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
                          options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 5: One more critical triangle polish after final NM
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        try:
            polished = _polish_critical_triangles(best_pts, max_iter=50, tol=1e-12)
            s = _score(polished)
            if s > best_score:
                best_score = s
                best_pts = polished
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
