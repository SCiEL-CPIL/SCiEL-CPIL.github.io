import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4.0

# Precompute indices for all 165 triangles
_triplets = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _triplets])
_T_J = np.array([t[1] for t in _triplets])
_T_K = np.array([t[2] for t in _triplets])

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _min_area_and_indices(points):
    """Return (min_area, list of indices of points involved in min triangles)."""
    areas = _compute_areas(points)
    min_area = float(np.min(areas))
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    involved = set()
    for idx in min_indices:
        involved.add(int(_T_I[idx]))
        involved.add(int(_T_J[idx]))
        involved.add(int(_T_K[idx]))
    return min_area, sorted(involved)

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    eps = 1e-10
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        # y >= 0
        if y < -eps:
            y = 0.0
        # y <= sqrt(3) * x  (left edge)
        if y > S3 * x + eps:
            # project onto line y = sqrt(3)*x
            # closest point on line y = m*x is (x + m*(y-mx))/(1+m^2), (my + (y-mx))/(1+m^2) with m=sqrt(3)
            # Actually simpler: barycentric projection
            pass
        # y <= sqrt(3) * (1-x)  (right edge)
        if y > S3 * (1 - x) + eps:
            pass
        
        # Use barycentric coordinates for robust projection
        # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
        # Barycentric: a = 1 - x - y/sqrt(3), b = y/sqrt(3), c = x - y/sqrt(3)
        # Wait, let's use standard barycentric for equilateral triangle
        # V0=(0,0), V1=(1,0), V2=(0.5, H)
        # For point (x,y):
        # a = 1 - x - y/S3  (weight for V0)
        # b = y/S3          (weight for V1) -- wait no
        # Let's derive: P = a*V0 + b*V1 + c*V2, a+b+c=1
        # x = b*1 + c*0.5 = b + 0.5c
        # y = c*H
        # So c = y/H = 2y/S3
        # b = x - 0.5c = x - y/S3
        # a = 1 - b - c = 1 - x + y/S3 - 2y/S3 = 1 - x - y/S3
        c = y / S3H  # = 2y/S3
        b = x - 0.5 * c
        a = 1.0 - b - c
        
        if a < 0:
            # Clip a to 0, renormalize b and c
            a = 0.0
            s = b + c
            if s > 0:
                b /= s
                c /= s
            else:
                b = 0.5
                c = 0.5
        elif b < 0:
            b = 0.0
            s = a + c
            if s > 0:
                a /= s
                c /= s
            else:
                a = 0.5
                c = 0.5
        elif c < 0:
            c = 0.0
            s = a + b
            if s > 0:
                a /= s
                b /= s
            else:
                a = 0.5
                b = 0.5
        
        # Convert back to cartesian
        points[m, 0] = b * 1.0 + c * 0.5
        points[m, 1] = c * S3H
    return points

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - log_sum / beta

def _local_nudge_search(points, max_iter=500, step_sizes=None, time_limit=None):
    """
    Local search that only moves points involved in the smallest triangles.
    """
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min = _min_area(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    if time_limit is None:
        time_limit = 60.0
    start_t = time.time()
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            min_area, involved = _min_area_and_indices(current_points)
            
            if min_area > best_min:
                best_min = min_area
                best_points = current_points.copy()
            
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                
                for _ in range(20):
                    dx = np.random.uniform(-step_size, step_size)
                    dy = np.random.uniform(-step_size, step_size)
                    
                    new_x = current_points[idx, 0] + dx
                    new_y = current_points[idx, 1] + dy
                    
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_points = _project_to_triangle(test_points)
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_points, best_min

def _nelder_mead_optimize(points, time_limit=10.0, beta=50.0):
    """
    Use Nelder-Mead to optimize the soft min area objective.
    """
    start_t = time.time()
    x0 = points.flatten()
    
    def _objective(params):
        points = params.reshape(N, 2)
        points = _project_to_triangle(points.copy())
        return -_soft_min_area(points, beta)
    
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        return points, _min_area(points)
    
    new_points = result.x.reshape(N, 2)
    new_points = _project_to_triangle(new_points)
    score = _min_area(new_points)
    
    if score > _min_area(points):
        return new_points, score
    else:
        return points, _min_area(points)

def _lbfgsb_optimize(points, beta=50.0, maxiter=250):
    """
    Use L-BFGS-B to optimize the soft min area objective.
    """
    x0 = points.flatten()
    
    def _objective(params):
        points = params.reshape(N, 2)
        points = _project_to_triangle(points.copy())
        return -_soft_min_area(points, beta)
    
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_points = result.x.reshape(N, 2)
        new_points = _project_to_triangle(new_points)
        score = _min_area(new_points)
        
        if score > _min_area(points):
            return new_points, score
        else:
            return points, _min_area(points)
    except Exception:
        return points, _min_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = -np.inf
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (deadline - margin)
    
    def _update_best(pts):
        nonlocal best_points, best_score
        area = _min_area(pts)
        if area > best_score:
            best_score = area
            best_points = pts.copy()
    
    # Initial symmetric configuration
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    init = _project_to_triangle(init)
    _update_best(init)
    
    # Phase A: Multi-start with annealed beta schedule
    # Generate diverse starting points
    bases = [init]
    for _ in range(12):
        pts = init + rng.normal(0, 0.25, size=(N, 2))
        pts = _project_to_triangle(pts)
        bases.append(pts)
    
    n_starts = 24
    for s in range(n_starts):
        if not _elapsed_ok(35):
            break
        
        base = bases[s % len(bases)]
        pts = base + rng.normal(0, 0.06, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)
        
        # Annealed beta schedule
        for beta in [12.0, 35.0, 90.0, 220.0, 500.0]:
            if not _elapsed_ok(35):
                break
            
            # Alternate between L-BFGS-B and Nelder-Mead
            if beta <= 90.0:
                pts, _ = _lbfgsb_optimize(pts, beta=beta, maxiter=250)
            else:
                pts, _ = _nelder_mead_optimize(pts, time_limit=min(8.0, deadline - time.time() - 10), beta=beta)
            
            _update_best(pts)
    
    # Phase B: Nelder-Mead local polish on the best configuration
    if best_points is not None and _elapsed_ok(8):
        for scale in [0.05, 0.02, 0.01]:
            if not _elapsed_ok(6):
                break
            for _ in range(6):
                if not _elapsed_ok(6):
                    break
                cand = best_points + rng.normal(0, scale, size=(N, 2))
                cand = _project_to_triangle(cand)
                
                def nm_obj(x):
                    p = x.reshape(-1, 2)
                    p = _project_to_triangle(p)
                    return -_min_area(p)
                
                try:
                    res = minimize(nm_obj, cand.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-11})
                    p = res.x.reshape(-1, 2)
                    p = _project_to_triangle(p)
                    _update_best(p)
                except Exception:
                    pass
    
    # Phase C: Local nudge search for fine-tuning
    if best_points is not None and _elapsed_ok(10):
        refined_points, refined_score = _local_nudge_search(
            best_points,
            max_iter=500,
            step_sizes=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
            time_limit=min(8.0, deadline - time.time() - 5)
        )
        _update_best(refined_points)
    
    # Final fine polish on the absolute best
    if best_points is not None and _elapsed_ok(4):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_min_area(p)
        
        try:
            res = minimize(nm_obj, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13})
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass
    
    # Fallback if no valid result found
    if best_points is None:
        best_points = init
    
    # Final projection to ensure validity
    best_points = _project_to_triangle(best_points)
    
    return best_points
