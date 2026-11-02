"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2

_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 145.0


def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
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


def _active_set(pts, tol=1.02):
    """Return indices of triangles with area <= tol * min_area."""
    areas = _all_triangle_areas(pts)
    min_a = np.min(areas)
    return np.where(areas <= tol * min_a)[0]


def _push_away_from_min(pts, min_idx, step=0.01):
    """
    For the minimum-area triangle, compute a direction that increases its area
    by moving its vertices apart, then take a constrained step.
    Returns new points.
    """
    i, j, k = _T_I[min_idx], _T_J[min_idx], _T_K[min_idx]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    # To increase area, move vertices apart from centroid
    centroid = (pi + pj + pk) / 3.0
    
    # Direction from centroid to each vertex
    dir_i = pi - centroid
    dir_j = pj - centroid
    dir_k = pk - centroid
    
    # Normalize
    ni = np.linalg.norm(dir_i)
    nj = np.linalg.norm(dir_j)
    nk = np.linalg.norm(dir_k)
    
    if ni < 1e-10:
        dir_i = np.array([0.5, 0.0])
        ni = 1.0
    if nj < 1e-10:
        dir_j = np.array([0.5, 0.0])
        nj = 1.0
    if nk < 1e-10:
        dir_k = np.array([0.5, 0.0])
        nk = 1.0
    
    dir_i /= ni
    dir_j /= nj
    dir_k /= nk
    
    new_pts = pts.copy()
    new_pts[i] += step * dir_i
    new_pts[j] += step * dir_j
    new_pts[k] += step * dir_k
    new_pts = _project_to_triangle(new_pts)
    return new_pts


def _coordinate_descent(pts, max_iter=200, tol=1e-10):
    """
    Coordinate descent: for each point, try small perturbations to maximize
    the minimum triangle area.
    """
    current_min = _min_triangle_area(pts)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for it in range(max_iter):
        improved = False
        for idx in range(N):
            best_local = current_min
            best_pos = pts[idx].copy()
            
            for step in step_sizes:
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_pts = pts.copy()
                        new_pts[idx] = best_pos + np.array([dx, dy])
                        new_pts = _project_to_triangle(new_pts)
                        new_min = _min_triangle_area(new_pts)
                        if new_min > best_local + tol:
                            best_local = new_min
                            best_pos = new_pts[idx].copy()
            
            if np.linalg.norm(best_pos - pts[idx]) > 1e-12:
                pts[idx] = best_pos
                current_min = _min_triangle_area(pts)
                improved = True
        
        if not improved:
            break
    
    return pts


def _largest_empty_triangle_heuristic(pts, max_iter=100):
    """
    Iteratively identify the minimum-area triangle, compute a replacement
    candidate from the convex hull of other points, and swap if it improves.
    """
    current_min = _min_triangle_area(pts)
    
    for it in range(max_iter):
        areas = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        min_a = areas[min_idx]
        
        i, j, k = _T_I[min_idx], _T_J[min_idx], _T_K[min_idx]
        
        # For each vertex of the min triangle, try to find a better replacement
        best_improvement = 0.0
        best_swap = None
        
        for vertex_idx in [i, j, k]:
            # Get all other points
            others = [pts[m] for m in range(N) if m != vertex_idx]
            others_arr = np.array(others)
            
            # Compute centroid of the convex hull of others
            # Simple approach: use the average of extreme points
            # Find the point farthest from the centroid of all others
            centroid_all = np.mean(others_arr, axis=0)
            
            # Try moving the vertex away from the min triangle's opposite side
            # Direction: from the midpoint of the opposite edge to the vertex
            if vertex_idx == i:
                opp_mid = (pts[j] + pts[k]) / 2.0
            elif vertex_idx == j:
                opp_mid = (pts[i] + pts[k]) / 2.0
            else:
                opp_mid = (pts[i] + pts[j]) / 2.0
            
            direction = pts[vertex_idx] - opp_mid
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm
            
            # Try several candidate positions
            for step in [0.02, 0.05, 0.1, 0.15]:
                new_pos = pts[vertex_idx] + step * direction
                new_pos = _project_to_triangle(new_pos.reshape(1, 2))[0]
                
                new_pts = pts.copy()
                new_pts[vertex_idx] = new_pos
                new_min = _min_triangle_area(new_pts)
                
                if new_min > current_min + 1e-10:
                    improvement = new_min - current_min
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (vertex_idx, new_pos)
        
        if best_swap is not None:
            vertex_idx, new_pos = best_swap
            pts[vertex_idx] = new_pos
            current_min = _min_triangle_area(pts)
        else:
            # Try pushing the min triangle vertices apart
            new_pts = _push_away_from_min(pts, min_idx, step=0.005)
            new_min = _min_triangle_area(new_pts)
            if new_min > current_min + 1e-10:
                pts = new_pts
                current_min = new_min
            else:
                break
    
    return pts


def _random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u + 0.5 * v
    y = v * S3H
    return np.array([x, y])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Structured initial configurations
    inits = []
    
    # Config 1: Corners + edge midpoints + interior
    init1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    inits.append(init1)
    
    # Config 2: More symmetric interior placement
    init2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, S3H * 0.5],
        [0.25, S3H * 0.5],
        [0.75, S3H * 0.5],
        [0.5, S3H * 0.25],
        [0.5, S3H * 0.75],
        [0.5, S3H * 0.45],
    ], dtype=float)
    inits.append(init2)
    
    # Config 3: Based on 3x3 grid + corners
    init3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [1/3, 0.0],
        [2/3, 0.0],
        [0.5, S3H/3],
        [0.25, S3H/4],
        [0.75, S3H/4],
        [0.5, S3H/2],
        [0.333, S3H/3],
        [0.667, S3H/3],
    ], dtype=float)
    inits.append(init3)
    
    for init in inits:
        _update_best(init)
    
    # Phase A: Multi-start with largest-empty-triangle heuristic + coordinate descent
    n_starts = 40
    bases = list(inits)
    for _ in range(20):
        base_idx = rng.integers(0, len(inits))
        base = inits[base_idx] + rng.normal(0, 0.2, size=(N, 2))
        bases.append(_project_to_triangle(base))
    for _ in range(15):
        rand_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())
    
    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        base = bases[s % len(bases)]
        pts = base + rng.normal(0, 0.03, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)
        
        # Apply largest empty triangle heuristic
        try:
            pts = _largest_empty_triangle_heuristic(pts, max_iter=50)
            _update_best(pts)
        except Exception:
            pass
        
        # Apply coordinate descent polish
        try:
            pts = _coordinate_descent(pts, max_iter=100)
            _update_best(pts)
        except Exception:
            pass
    
    # Phase B: Nelder-Mead polish with exact min on best candidates
    if best_pts is not None and _elapsed_ok(25):
        bc_best = best_pts.copy()
        
        for scale in [0.03, 0.015, 0.008, 0.004]:
            if not _elapsed_ok(15):
                break
            for _ in range(8):
                if not _elapsed_ok(10):
                    break
                pert = bc_best + rng.normal(0, scale, size=(N, 2))
                pert = _project_to_triangle(pert)
                
                def nm_obj(x):
                    px = _project_to_triangle(x.reshape(-1, 2))
                    return -_min_triangle_area(px)
                
                try:
                    res = minimize(nm_obj, pert.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 6000, 'xatol': 1e-12, 'fatol': 1e-14})
                    p = _project_to_triangle(res.x.reshape(-1, 2))
                    _update_best(p)
                    if _min_triangle_area(p) > _min_triangle_area(bc_best):
                        bc_best = p.copy()
                except Exception:
                    pass
    
    # Phase C: Final ultra-fine polish
    if best_pts is not None and _elapsed_ok(10):
        flat = best_pts.flatten()
        
        def final_nm(x):
            px = _project_to_triangle(x.reshape(-1, 2))
            return -_min_triangle_area(px)
        
        try:
            res = minimize(final_nm, flat, method='Nelder-Mead',
                           options={'maxiter': 12000, 'xatol': 1e-14, 'fatol': 1e-16})
            p = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(p)
        except Exception:
            pass
        
        # Final coordinate descent
        try:
            p = _coordinate_descent(best_pts, max_iter=200)
            _update_best(p)
        except Exception:
            pass
    
    # Phase D: Coordinate descent on active set
    if best_pts is not None and _elapsed_ok(5):
        pts = best_pts.copy()
        for _ in range(50):
            if not _elapsed_ok(3):
                break
            areas = _all_triangle_areas(pts)
            min_a = np.min(areas)
            active = set()
            for i, a in enumerate(areas):
                if a < min_a * 1.01:
                    active.add(_T_I[i])
                    active.add(_T_J[i])
                    active.add(_T_K[i])
            
            improved = False
            for pt_idx in active:
                for _ in range(3):
                    if not _elapsed_ok(2):
                        break
                    new_pts = pts.copy()
                    new_pts[pt_idx] += rng.normal(0, 0.005, size=2)
                    new_pts = _project_to_triangle(new_pts)
                    if _min_triangle_area(new_pts) > min_a:
                        pts = new_pts
                        improved = True
                        min_a = _min_triangle_area(pts)
                        break
            if not improved:
                break
        _update_best(pts)
    
    if best_pts is None:
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        return pts
    
    return _project_to_triangle(best_pts)
