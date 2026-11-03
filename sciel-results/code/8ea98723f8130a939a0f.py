"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Combined strategy:
  1. Fast vectorized triangle area computation.
  2. Directed push-pull local search: identify worst triangle, push vertices
     outward along the triangle's outward normal to increase its area.
  3. Multi-phase optimization: Nelder-Mead on Cartesian space, then Powell,
     L-BFGS-B with soft objective, differential evolution for global search.
  4. Centrally symmetric and ring parameterizations for structured search.
  5. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(flat, beta=80.0):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


def _push_pull_search(pts, deadline, max_iter=2000):
    """
    Directed push-pull local search:
    - Find the worst (smallest area) triangle.
    - Push its vertices outward along the direction that increases the triangle area.
    - Use line search to find the best step.
    """
    pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        # Compute all triangle areas
        areas = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        if min_area < 1e-12:
            break
        
        # Get the worst triangle vertices
        i, j, k = tri_indices[min_idx]
        
        # Compute the outward normal direction for each vertex to increase area
        # For triangle (i, j, k), the area is 0.5 * |(pj - pi) x (pk - pi)|
        # To increase area, we can push vertices outward from the centroid
        
        centroid = (pts[i] + pts[j] + pts[k]) / 3.0
        
        # Directions to push each vertex
        dirs = []
        for v in [i, j, k]:
            d = pts[v] - centroid
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                d = d / norm
            else:
                d = np.array([1.0, 0.0])
            dirs.append(d)
        
        # Try pushing each vertex in its direction with different step sizes
        best_step = 0.0
        best_new_score = best_score
        
        for v_idx, v in enumerate([i, j, k]):
            d = dirs[v_idx]
            for step in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
                pts_new = pts.copy()
                pts_new[v] += step * d
                s = _score(pts_new)
                if s > best_new_score:
                    best_new_score = s
                    best_step = step
                    best_v = v
                    best_d = d
        
        if best_step > 0:
            pts[best_v] += best_step * best_d
            best_score = best_new_score
        
        # Also try small random perturbations to escape local minima
        if it % 50 == 0:
            for _ in range(5):
                pts_pert = pts + np.random.normal(0, 0.01, size=pts.shape)
                s = _score(pts_pert)
                if s > best_score:
                    pts = pts_pert
                    best_score = s
    
    return pts


def _coordinate_descent(pts, deadline, max_rounds=30):
    """Cyclic coordinate descent on the hard objective."""
    pts = pts.copy()
    n = len(pts)
    step0 = 0.05
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(25):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    if s_plus > s_minus:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.5
                    if step < 1e-8:
                        break
        if not improved:
            break
    return pts


def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points


def _make_cartesian_inits():
    """Generate diverse initial point configurations (Cartesian)."""
    inits = []
    rng = np.random.default_rng(42)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Mixed ring configurations
    for r1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts)
    
    # 3-ring configurations
    for r1, r2 in [(0.2, 0.5), (0.3, 0.6), (0.4, 0.7)]:
        a1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        inits.append(pts)
    
    # Random points
    for seed in range(15):
        rs = np.random.default_rng(seed * 37 + 19)
        pts = rs.uniform(-1, 1, size=(N, 2))
        inits.append(pts)
    
    # Perturbed regular polygon
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.15, 0.2, 0.3]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))
    
    # Elliptical
    for rx, ry in [(1.0, 0.6), (1.0, 0.7), (1.0, 0.8), (1.2, 0.7), (0.8, 1.2), (1.3, 0.65)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))
    
    return inits


def _nm(flat0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def _lbfgs(flat0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, flat0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_hard(pts.flatten())
    
    # Phase 1: Generate diverse initial configurations
    inits = _make_cartesian_inits()
    
    # Phase 2: Nelder-Mead from multiple starts
    for i, init in enumerate(inits):
        if time.time() > deadline - 60:
            break
        res = _nm(init.flatten(), deadline, _objective_hard, max_iter=1500)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 3: Push-pull search from best found
    if best_pts is not None and time.time() < deadline - 40:
        pp_pts = _push_pull_search(best_pts, deadline, max_iter=2000)
        update_best(pp_pts)
    
    # Phase 4: More Nelder-Mead from best after push-pull
    if best_pts is not None and time.time() < deadline - 30:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=2500)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 5: Push-pull again
    if best_pts is not None and time.time() < deadline - 25:
        pp_pts = _push_pull_search(best_pts, deadline, max_iter=2000)
        update_best(pp_pts)
    
    # Phase 6: L-BFGS-B with soft objective
    if best_pts is not None and time.time() < deadline - 20:
        for beta in [50.0, 80.0, 120.0, 150.0]:
            if time.time() > deadline - 15:
                break
            res = _lbfgs(best_pts.flatten(), deadline, lambda x, b=beta: _objective_soft(x, b), max_iter=3000)
            if res is not None and res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
    
    # Phase 7: Powell refinement
    if best_pts is not None and time.time() < deadline - 15:
        res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=2000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 8: Push-pull after Powell
    if best_pts is not None and time.time() < deadline - 12:
        pp_pts = _push_pull_search(best_pts, deadline, max_iter=1500)
        update_best(pp_pts)
    
    # Phase 9: Differential Evolution (Global Search)
    if time.time() < deadline - 12:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            de_res = differential_evolution(
                _objective_hard, bounds,
                seed=42, maxiter=50, tol=1e-8,
                popsize=12, mutation=(0.4, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                update_best(de_res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 10: Coordinate descent refinement
    if best_pts is not None and time.time() < deadline - 8:
        cd_pts = _coordinate_descent(best_pts, deadline, max_rounds=20)
        update_best(cd_pts)
    
    # Phase 11: Final push-pull
    if best_pts is not None and time.time() < deadline - 5:
        pp_pts = _push_pull_search(best_pts, deadline, max_iter=1000)
        update_best(pp_pts)
    
    # Phase 12: Final Polish with tighter tolerances
    if best_pts is not None and time.time() < deadline - 3:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=5000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
        
        if time.time() < deadline - 2:
            res = _powell(best_pts.flatten(), deadline, _objective_hard, max_iter=3000)
            if res is not None and res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        
        if time.time() < deadline - 1:
            # One more NM pass with very tight tolerance
            res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=5000)
            if res is not None and res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
