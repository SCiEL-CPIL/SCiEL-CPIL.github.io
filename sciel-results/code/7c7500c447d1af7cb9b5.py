import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute all triplets of indices
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
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
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points, ensuring validity."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    eps = 1e-10
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < -eps:
            points[m, 1] = 0.0
        if y + SQRT3 * x > SQRT3 + eps:
            d = (y + SQRT3 * x - SQRT3) / 2.0
            if d > 0:
                points[m, 0] -= SQRT3 * d
                points[m, 1] -= d
        if y > SQRT3 * x + eps:
            d = (SQRT3 * x - y) / 2.0
            if d < 0:
                points[m, 0] += SQRT3 * d
                points[m, 1] -= d
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < 0:
            points[m, 1] = 0
        if y > SQRT3 * x:
            if x > 0:
                points[m, 1] = SQRT3 * x
            else:
                points[m, 1] = 0
        if y > SQRT3 * (1 - x):
            if x < 1:
                points[m, 1] = SQRT3 * (1 - x)
            else:
                points[m, 1] = 0
    return points

def _sobol_points_in_triangle(n, seed=42):
    """Generate n quasi-random points in the equilateral triangle using Sobol sequence."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    n_gen = n * 4
    samples = rng.random(n_gen)
    u = samples[:, 0]
    v = samples[:, 1]
    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    valid_u = u[mask]
    valid_v = v[mask]
    if len(valid_u) >= n:
        u = valid_u[:n]
        v = valid_v[:n]
    else:
        while len(valid_u) < n:
            more = rng.random(n_gen)
            u2, v2 = more[:, 0], more[:, 1]
            mask2 = (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1)
            valid_u = np.concatenate([valid_u, u2[mask2]])
            valid_v = np.concatenate([valid_v, v2[mask2]])
        u = valid_u[:n]
        v = valid_v[:n]
    bary = np.column_stack([u, v])
    return _bary_to_cart(bary)

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    -log(sum(exp(-beta * areas))) / beta approximates min(areas) for large beta.
    """
    areas = _compute_areas(points)
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    # log(sum(exp(-beta * (areas - min_a))))
    # sum(exp(-beta * (areas - min_a))) = 1 + sum(exp(-beta * (areas - min_a))) for i where areas[i] != min_a
    # Actually simpler:
    # log_sum = log(sum(exp(-beta * areas)))
    # log_sum = log(exp(-beta * min_a) * sum(exp(-beta * (areas - min_a))))
    # log_sum = -beta * min_a + log(sum(exp(-beta * (areas - min_a))))
    # min_soft = -log_sum / beta = min_a - log(sum(exp(-beta * (areas - min_a)))) / beta
    
    shifted = -beta * (areas - min_a)
    # To avoid overflow/underflow, we can just compute directly since areas-min_a >= 0, so shifted <= 0
    # exp(shifted) <= 1, so sum is at most 165, log is safe.
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_a - log_sum / beta

def _objective(params, beta=50.0):
    """
    Objective function for optimization: negative of soft min area.
    params is a flat array of 22 coordinates (x0,y0,x1,y1,...,x10,y10).
    """
    points = params.reshape(N, 2)
    # Project to triangle to handle constraint violations
    points = _project_to_triangle(points.copy())
    return -_soft_min_area(points, beta)

def _nelder_mead_optimize(points, time_limit=10.0, beta=50.0):
    """
    Use Nelder-Mead to optimize the soft min area objective.
    """
    start_t = time.time()
    x0 = points.flatten()
    
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        args=(beta,),
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

def _local_nudge_search(points, max_iter=500, step_sizes=None, time_limit=None):
    """
    Local search that only moves points involved in the smallest triangles.
    """
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min, _ = _min_area_and_indices(current_points)
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
                    du = np.random.uniform(-step_size, step_size)
                    dv = np.random.uniform(-step_size, step_size)
                    
                    bary = _cart_to_bary(current_points)
                    new_u = bary[idx, 0] + du
                    new_v = bary[idx, 1] + dv
                    
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                    
                    new_x = new_v * 1.0 + (1.0 - new_u - new_v) * 0.5
                    new_y = (1.0 - new_u - new_v) * H
                    
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                    
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        improved = True
                        break
            
            if not improved:
                break
    
    return best_points, best_min

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_points = None
    best_val = -np.inf
    
    # 1. Initializations
    inits = []
    
    # Sobol
    try:
        inits.append(_sobol_points_in_triangle(N, seed=42))
        inits.append(_sobol_points_in_triangle(N, seed=123))
    except Exception:
        pass
    
    # Random
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v <= 1:
                    break
            b = np.array([u, v])
            pts[i] = _bary_to_cart(b.reshape(1, 2))[0]
        inits.append(pts)
    
    # Grid-like (corners and midpoints)
    # Corners: (0,0), (1,0), (0.5, H)
    # Midpoints: (0.5, 0), (0.25, H/2), (0.75, H/2)
    # Center: (0.5, H/3)
    # This is a heuristic start
    grid_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, 0.0],
        [0.25, H/2],
        [0.75, H/2],
        [0.5, H/3],
        [0.25, H/6],
        [0.75, H/6],
        [0.5, H/2],
        [0.25, H/4]
    ])
    inits.append(grid_pts)
    
    # 2. Optimize each initialization
    for init in inits:
        if time.time() > deadline - 20:
            break
        
        # Nelder-Mead with soft min
        try:
            opt_pts, opt_val = _nelder_mead_optimize(init, time_limit=10.0, beta=50.0)
            if opt_val > best_val:
                best_val = opt_val
                best_points = opt_pts
        except Exception:
            pass
        
        # Local nudge search
        if time.time() > deadline - 10:
            break
        try:
            nudge_pts, nudge_val = _local_nudge_search(opt_pts, max_iter=200, time_limit=10.0)
            if nudge_val > best_val:
                best_val = nudge_val
                best_points = nudge_pts
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        best_points = inits[0] if inits else np.zeros((N, 2))
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            # Increase beta for sharper minimum
            final_pts, final_val = _nelder_mead_optimize(best_points, time_limit=5.0, beta=100.0)
            if final_val > best_val:
                best_points = final_pts
                best_val = final_val
        except Exception:
            pass
    
    return best_points

