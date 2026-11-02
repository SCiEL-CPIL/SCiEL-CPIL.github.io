"""Heilbronn 11 triangles: Direct Cartesian optimization with multi-start and local polish."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, S3H])


def _project_to_triangle(pts):
    """Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_triangle_areas(pts):
    """Compute all triangle areas."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _push_apart(pts, k=5):
    """
    Direct 'push apart' strategy: for the k smallest triangles, 
    move points apart to increase their areas.
    """
    areas = _all_triangle_areas(pts)
    # Get indices of k smallest areas
    k = min(k, len(areas))
    small_idx = np.argsort(areas)[:k]
    
    pts_new = pts.copy()
    for idx in small_idx:
        i, j, k_idx = _TRIPLETS[idx]
        pi = pts_new[i]
        pj = pts_new[j]
        pk = pts_new[k_idx]
        
        # Compute triangle centroid
        centroid = (pi + pj + pk) / 3
        
        # Direction to move each point: away from centroid
        # But we need to be careful not to push points outside the triangle
        # Use small step size
        step = 0.005
        
        for p_idx, p in [(i, pi), (j, pj), (k_idx, pk)]:
            direction = p - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                pts_new[p_idx] = p + step * direction
    
    # Project back to triangle
    pts_new = _project_to_triangle(pts_new)
    return pts_new


def _objective(x):
    """Objective function for optimization (negated min area)."""
    p = x.reshape(-1, 2)
    p = _project_to_triangle(p)
    return -_min_triangle_area(p)


def _objective_soft(x, beta):
    """Soft objective function for optimization."""
    p = x.reshape(-1, 2)
    p = _project_to_triangle(p)
    return -_soft_min_area(p, beta)


def _generate_initial_configs(rng):
    """Generate diverse starting configurations."""
    configs = []
    
    # 1. Symmetric base configuration
    base_sym = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [1/6, S3H/6],
        [5/6, S3H/6],
        [1/3, 2*S3H/3],
        [2/3, 2*S3H/3],
    ])
    configs.append(base_sym.copy())
    
    # 2. Random configurations
    for _ in range(10):
        pts = np.random.random((N, 2))
        # Map to triangle
        for i in range(N):
            u, v = pts[i]
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        configs.append(_project_to_triangle(pts))
    
    # 3. Perturbed symmetric base
    for _ in range(5):
        perturbed = base_sym + rng.normal(0, 0.05, size=(N, 2))
        configs.append(_project_to_triangle(perturbed))
    
    return configs


def _local_optimize(pts, max_time=10):
    """Local optimization using multiple methods."""
    start_time = time.time()
    best_pts = pts.copy()
    best_area = _min_triangle_area(pts)
    
    # Method 1: Nelder-Mead
    if time.time() - start_time < max_time - 5:
        try:
            res = minimize(
                _objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            area = _min_triangle_area(p)
            if area > best_area:
                best_area = area
                best_pts = p.copy()
        except Exception:
            pass
    
    # Method 2: Powell
    if time.time() - start_time < max_time - 3:
        try:
            res = minimize(
                _objective,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 100, 'maxfev': 3000, 'xrtol': 1e-10, 'fatol': 1e-12}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            area = _min_triangle_area(p)
            if area > best_area:
                best_area = area
                best_pts = p.copy()
        except Exception:
            pass
    
    # Method 3: Push-apart iterations
    if time.time() - start_time < max_time - 2:
        for _ in range(50):
            if time.time() - start_time > max_time - 1:
                break
            pts_pushed = _push_apart(best_pts, k=5)
            area = _min_triangle_area(pts_pushed)
            if area > best_area:
                best_area = area
                best_pts = pts_pushed.copy()
            else:
                break
    
    return best_pts, best_area


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts)
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Phase 1: Multi-start with local optimization
    configs = _generate_initial_configs(rng)
    
    for config in configs:
        if not _elapsed_ok(15):
            break
        
        # Local optimize each config
        optimized_pts, optimized_area = _local_optimize(config, max_time=8)
        _update_best(optimized_pts)
    
    # Phase 2: Iterative push-apart + local polish on best
    if best_pts is not None and _elapsed_ok(30):
        current_pts = best_pts.copy()
        for iteration in range(20):
            if not _elapsed_ok(5):
                break
            
            # Push apart
            pts_pushed = _push_apart(current_pts, k=8)
            area_pushed = _min_triangle_area(pts_pushed)
            
            if area_pushed > _min_triangle_area(current_pts):
                current_pts = pts_pushed
                _update_best(current_pts)
            
            # Local polish
            if _elapsed_ok(3):
                polished_pts, polished_area = _local_optimize(current_pts, max_time=5)
                _update_best(polished_pts)
                current_pts = polished_pts
    
    # Phase 3: Simulated annealing for global exploration
    if best_pts is not None and _elapsed_ok(40):
        current_pts = best_pts.copy()
        current_area = _min_triangle_area(current_pts)
        T_start = 0.05
        T_end = 0.001
        n_iter = 3000
        
        for i in range(n_iter):
            if not _elapsed_ok(5):
                break
            
            T = T_start * (T_end / T_start) ** (i / n_iter)
            
            # Perturb
            perturbation = rng.normal(0, 0.02, size=(N, 2))
            new_pts = current_pts + perturbation
            new_pts = _project_to_triangle(new_pts)
            new_area = _min_triangle_area(new_pts)
            
            # Acceptance
            if new_area >= current_area:
                current_pts = new_pts
                current_area = new_area
                _update_best(current_pts)
            else:
                delta = new_area - current_area
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_area = new_area
        
        # Final polish after SA
        if _elapsed_ok(10):
            polished_pts, polished_area = _local_optimize(current_pts, max_time=8)
            _update_best(polished_pts)
    
    # Phase 4: Final fine polish
    if best_pts is not None and _elapsed_ok(10):
        polished_pts, polished_area = _local_optimize(best_pts, max_time=8)
        _update_best(polished_pts)
    
    if best_pts is None:
        # Fallback: symmetric configuration
        best_pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, S3H],
            [0.5, 0.0],
            [0.75, S3H / 2],
            [0.25, S3H / 2],
            [1/3, S3H/3],
            [1/6, S3H/6],
            [5/6, S3H/6],
            [1/3, 2*S3H/3],
            [2/3, 2*S3H/3],
        ])
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts
