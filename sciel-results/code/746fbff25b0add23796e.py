"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
DEADLINE_FRACTION = 0.85  # safety margin

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _in_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    x, y = p
    # Barycentric coordinates
    d0 = (V1[0] - V0[0]) * (y - V0[1]) - (V1[1] - V0[1]) * (x - V0[0])
    d1 = (V2[0] - V1[0]) * (y - V1[1]) - (V2[1] - V1[1]) * (x - V1[0])
    d2 = (V0[0] - V2[0]) * (y - V2[1]) - (V0[1] - V2[1]) * (x - V2[0])
    # All must have same sign (or zero)
    has_neg = (d0 < -1e-12) or (d1 < -1e-12) or (d2 < -1e-12)
    has_pos = (d0 > 1e-12) or (d1 > 1e-12) or (d2 > 1e-12)
    return not (has_neg and has_pos)


def _project_to_triangle(p):
    """Project point p onto the triangle (closest point on or inside the triangle)."""
    if _in_triangle(p):
        return p.copy()
    
    # Find closest point on the triangle boundary
    def dist_to_segment(a, b, p):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        return np.linalg.norm(p - proj), proj
    
    best_dist = np.inf
    best_proj = p.copy()
    for i in range(3):
        a = [V0, V1, V2][i]
        b = [V0, V1, V2][(i + 1) % 3]
        d, proj = dist_to_segment(a, b, p)
        if d < best_dist:
            best_dist = d
            best_proj = proj
    return best_proj


def _flatten_points(points):
    """Convert (n,2) array to flat 2n vector."""
    return points.flatten()


def _unflatten_points(x):
    """Convert flat 2n vector to (n,2) array."""
    return x.reshape(-1, 2)


def _penalty_objective(x, lam=100.0):
    """
    Objective to minimize: -min_triangle_area + penalty for points outside triangle.
    """
    points = _unflatten_points(x)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for p in points:
        if not _in_triangle(p):
            # Distance to triangle boundary
            def dist_to_segment(a, b, p):
                ab = b - a
                t = np.dot(p - a, ab) / np.dot(ab, ab)
                t = np.clip(t, 0, 1)
                proj = a + t * ab
                return np.linalg.norm(p - proj)
            
            min_d = np.inf
            for i in range(3):
                a = [V0, V1, V2][i]
                b = [V0, V1, V2][(i + 1) % 3]
                d = dist_to_segment(a, b, p)
                if d < min_d:
                    min_d = d
            penalty += lam * min_d ** 2
    
    min_area = _min_triangle_area(points)
    return -min_area + penalty


def _logsumexp_surrogate(x, beta=50.0):
    """
    Smooth surrogate: -logsumexp(-beta * log(areas)) approximates -min(areas) for large beta.
    We minimize this.
    """
    points = _unflatten_points(x)
    n = len(points)
    
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    
    areas = np.array(areas)
    # Avoid log(0)
    areas = np.maximum(areas, 1e-15)
    
    log_areas = np.log(areas)
    max_log = np.max(log_areas)
    lse = max_log + np.log(np.sum(np.exp(log_areas - max_log)))
    
    # Penalty for points outside triangle
    penalty = 0.0
    for p in points:
        if not _in_triangle(p):
            def dist_to_segment(a, b, p):
                ab = b - a
                t = np.dot(p - a, ab) / np.dot(ab, ab)
                t = np.clip(t, 0, 1)
                proj = a + t * ab
                return np.linalg.norm(p - proj)
            
            min_d = np.inf
            for i in range(3):
                a = [V0, V1, V2][i]
                b = [V0, V1, V2][(i + 1) % 3]
                d = dist_to_segment(a, b, p)
                if d < min_d:
                    min_d = d
            penalty += 100.0 * min_d ** 2
    
    return -lse + penalty


def _generate_structured_init(rng, n=N):
    """Generate structured initial points: grid-like and symmetric patterns."""
    points = []
    
    # Include vertices
    points.append(V0.copy())
    points.append(V1.copy())
    points.append(V2.copy())
    
    # Add midpoints of edges
    points.append((V0 + V1) / 2.0)
    points.append((V1 + V2) / 2.0)
    points.append((V2 + V0) / 2.0)
    
    # Add centroid
    points.append((V0 + V1 + V2) / 3.0)
    
    # Add some interior points in a hexagonal-ish pattern
    cx, cy = (V0[0] + V1[0] + V2[0]) / 3.0, (V0[1] + V1[1] + V2[1]) / 3.0
    r = 0.15
    for angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        if _in_triangle([px, py]):
            points.append(np.array([px, py]))
    
    # Trim or pad to exactly n points
    points = points[:n]
    while len(points) < n:
        # Random point inside triangle
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        p = V0 + u * (V1 - V0) + v * (V2 - V0)
        points.append(p)
    
    return np.array(points[:n])


def _generate_random_init(rng, n=N):
    """Generate random initial points inside the triangle."""
    points = []
    for _ in range(n):
        u = rng.random()
        v = rng.random()
        while u + v > 1.0:
            u = rng.random()
            v = rng.random()
        p = V0 + u * (V1 - V0) + v * (V2 - V0)
        points.append(p)
    return np.array(points)


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_area = -np.inf
    
    def _evaluate_and_update(points):
        nonlocal best_points, best_area
        # Ensure all points are inside triangle
        for i in range(len(points)):
            if not _in_triangle(points[i]):
                points[i] = _project_to_triangle(points[i])
        
        area = _min_triangle_area(points)
        if area > best_area:
            best_area = area
            best_points = points.copy()
        return area
    
    # Generate multiple diverse starting points
    n_starts = 8
    starts = []
    
    # Structured initializations
    for seed in range(3):
        r = np.random.default_rng(seed * 100 + 1)
        starts.append(_generate_structured_init(r))
    
    # Random initializations
    for seed in range(5):
        r = np.random.default_rng(seed * 200 + 7)
        starts.append(_generate_random_init(r))
    
    # Evaluate all starts
    for i, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
        _evaluate_and_update(start_pts)
    
    # Local optimization from each start
    for i, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
        
        x0 = _flatten_points(start_pts)
        
        # Use trust-reflective with logsumexp surrogate
        try:
            res = minimize(
                _logsumexp_surrogate,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < _logsumexp_surrogate(x0):
                pts = _unflatten_points(res.x)
                # Project to triangle
                for j in range(len(pts)):
                    if not _in_triangle(pts[j]):
                        pts[j] = _project_to_triangle(pts[j])
                _evaluate_and_update(pts)
        except Exception:
            pass
        
        # Nelder-Mead refinement on exact objective
        if time.time() > deadline:
            break
        
        try:
            x0_nm = _flatten_points(start_pts)
            
            def nm_obj(x):
                points = _unflatten_points(x)
                # Penalty for outside points
                pen = 0.0
                for p in points:
                    if not _in_triangle(p):
                        def dist_to_segment(a, b, p):
                            ab = b - a
                            t = np.dot(p - a, ab) / np.dot(ab, ab)
                            t = np.clip(t, 0, 1)
                            proj = a + t * ab
                            return np.linalg.norm(p - proj)
                        min_d = np.inf
                        for ii in range(3):
                            a = [V0, V1, V2][ii]
                            b = [V0, V1, V2][(ii + 1) % 3]
                            d = dist_to_segment(a, b, p)
                            if d < min_d:
                                min_d = d
                        pen += 50.0 * min_d ** 2
                area = _min_triangle_area(points)
                return -area + pen
            
            res_nm = minimize(
                nm_obj,
                x0_nm,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            pts = _unflatten_points(res_nm.x)
            for j in range(len(pts)):
                if not _in_triangle(pts[j]):
                    pts[j] = _project_to_triangle(pts[j])
            _evaluate_and_update(pts)
        except Exception:
            pass
    
    # If we have time, do differential evolution for global search
    if time.time() < deadline - 30:
        try:
            # Define bounds: all points in [0,1]x[0,1], then project
            bounds = [(0, 1)] * (2 * N)
            
            def de_obj(x):
                points = _unflatten_points(x)
                pen = 0.0
                for p in points:
                    if not _in_triangle(p):
                        def dist_to_segment(a, b, p):
                            ab = b - a
                            t = np.dot(p - a, ab) / np.dot(ab, ab)
                            t = np.clip(t, 0, 1)
                            proj = a + t * ab
                            return np.linalg.norm(p - proj)
                        min_d = np.inf
                        for ii in range(3):
                            a = [V0, V1, V2][ii]
                            b = [V0, V1, V2][(ii + 1) % 3]
                            d = dist_to_segment(a, b, p)
                            if d < min_d:
                                min_d = d
                        pen += 50.0 * min_d ** 2
                area = _min_triangle_area(points)
                return -area + pen
            
            de_res = differential_evolution(
                de_obj,
                bounds,
                maxiter=50,
                popsize=15,
                seed=123,
                tol=1e-8,
                polish=True,
                init='sobol'
            )
            
            pts = _unflatten_points(de_res.x)
            for j in range(len(pts)):
                if not _in_triangle(pts[j]):
                    pts[j] = _project_to_triangle(pts[j])
            _evaluate_and_update(pts)
        except Exception:
            pass
    
    # Final Nelder-Mead refinement from best found
    if best_points is not None and time.time() < deadline - 5:
        try:
            x0 = _flatten_points(best_points)
            
            def nm_obj2(x):
                points = _unflatten_points(x)
                pen = 0.0
                for p in points:
                    if not _in_triangle(p):
                        def dist_to_segment(a, b, p):
                            ab = b - a
                            t = np.dot(p - a, ab) / np.dot(ab, ab)
                            t = np.clip(t, 0, 1)
                            proj = a + t * ab
                            return np.linalg.norm(p - proj)
                        min_d = np.inf
                        for ii in range(3):
                            a = [V0, V1, V2][ii]
                            b = [V0, V1, V2][(ii + 1) % 3]
                            d = dist_to_segment(a, b, p)
                            if d < min_d:
                                min_d = d
                        pen += 100.0 * min_d ** 2
                area = _min_triangle_area(points)
                return -area + pen
            
            res_final = minimize(
                nm_obj2,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            pts = _unflatten_points(res_final.x)
            for j in range(len(pts)):
                if not _in_triangle(pts[j]):
                    pts[j] = _project_to_triangle(pts[j])
            _evaluate_and_update(pts)
        except Exception:
            pass
    
    # Return best points found
    if best_points is not None:
        return best_points
    else:
        # Fallback: return structured init
        r = np.random.default_rng(0)
        return _generate_structured_init(r)
