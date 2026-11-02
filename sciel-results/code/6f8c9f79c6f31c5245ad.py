import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
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
    
    # Area = 0.5 * |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
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
    # Find all triangles with area close to min
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    # Collect unique point indices involved
    involved = set()
    for idx in min_indices:
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    -log(sum(exp(-beta * areas))) / beta approximates min(areas) for large beta.
    """
    areas = _compute_areas(points)
    # Use log-sum-exp trick for numerical stability
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta

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
        # u + v = 1 - y/H
        # v - u = 2*x - 1
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

def _local_nudge_search(points, max_iter=500, step_sizes=None, time_limit=10.0, rng=None):
    """
    Local search that only moves points involved in the smallest triangles.
    Uses targeted perturbations to break near-degeneracies.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min, _ = _min_area_and_indices(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
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
                
            if len(involved) == 0:
                break
                
            # Try nudging each involved point in random directions
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                    
                # Try several random perturbations for this point
                for _ in range(20):
                    du = rng.uniform(-step_size, step_size)
                    dv = rng.uniform(-step_size, step_size)
                    
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
                    
                    # Check if inside triangle
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                        
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        min_area = test_min
                        improved = True
                        break
                        
                if improved:
                    break
                    
            if not improved:
                break
                
    return best_points, best_min

def _initial_guess_grid():
    """Generate a structured initial guess based on a grid pattern."""
    positions = [
        (0.1, 0.1), (0.4, 0.1), (0.7, 0.1),
        (0.2, 0.3), (0.5, 0.3), (0.8, 0.3),
        (0.15, 0.5), (0.45, 0.5), (0.75, 0.5),
        (0.3, 0.7), (0.6, 0.7),
    ]
    bary = np.empty((N, 2))
    for m, (u, v) in enumerate(positions):
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        bary[m, 0] = u
        bary[m, 1] = v
    return bary

def _random_valid_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.empty((N, 2))
    for m in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[m, 0] = u
                bary[m, 1] = v
                break
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    def _try_nm_optimize(points, beta, time_limit):
        """Try to optimize from initial points using Nelder-Mead, return (points, score) or (None, -inf)."""
        try:
            pts, score = _nelder_mead_optimize(points, time_limit=time_limit, beta=beta)
            # Verify validity
            pts = _project_to_triangle(pts.copy())
            if np.all(pts[:, 0] >= -1e-8) and np.all(pts[:, 0] <= 1.0 + 1e-8) and \
               np.all(pts[:, 1] >= -1e-8) and np.all(pts[:, 1] <= H + 1e-8):
                x = pts[:, 0]
                y = pts[:, 1]
                if np.all(y >= -1e-8) and np.all(y <= SQRT3 * x + 1e-8) and \
                   np.all(y <= -SQRT3 * x + SQRT3 + 1e-8):
                    score = _min_area(pts)
                    return pts, score
        except Exception:
            pass
        return None, -np.inf
    
    # Phase 1: Multiple restarts with Nelder-Mead using smooth objective
    
    # Start with a good structured guess
    initial_bary = _initial_guess_grid()
    initial_pts = _bary_to_cart(initial_bary)
    
    pts, score = _try_nm_optimize(initial_pts, beta=50.0, time_limit=15.0)
    if pts is not None and score > best_score:
        best_points = pts
        best_score = score
        
    # Apply local nudge search to the initial result
    if best_points is not None and time.time() < deadline - 40:
        nudged_pts, nudged_score = _local_nudge_search(best_points, max_iter=300, time_limit=15.0, rng=rng)
        if nudged_score > best_score:
            best_points = nudged_pts
            best_score = nudged_score

    # Additional restarts from random and Sobol initial points
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Alternate between random and Sobol initial points
        if i % 2 == 0:
            random_bary = _random_valid_barycentric(rng)
            random_pts = _bary_to_cart(random_bary)
        else:
            random_pts = _sobol_points_in_triangle(N, seed=42 + i)
        
        # Use slightly different betas
        beta = 50.0 + 10.0 * i
        
        pts, score = _try_nm_optimize(random_pts, beta=beta, time_limit=10.0)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
            
        # Apply local nudge search to each random restart result
        if pts is not None and time.time() < deadline - 20:
            nudged_pts, nudged_score = _local_nudge_search(pts, max_iter=200, time_limit=5.0, rng=rng)
            if nudged_score > best_score:
                best_points = nudged_pts
                best_score = nudged_score
    
    # Phase 2: Refine best result with higher beta for higher precision
    if best_points is not None:
        if time.time() > deadline - 15:
            return best_points
        
        # Refine with higher beta
        pts, score = _try_nm_optimize(best_points, beta=100.0, time_limit=15.0)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
            
        # Apply nudge search to refined result
        if time.time() < deadline - 10:
            nudged_pts, nudged_score = _local_nudge_search(best_points, max_iter=400, time_limit=10.0, rng=rng)
            if nudged_score > best_score:
                best_points = nudged_pts
                best_score = nudged_score

        # Further refinement with even higher beta
        if time.time() > deadline - 10:
            return best_points
        
        pts, score = _try_nm_optimize(best_points, beta=200.0, time_limit=10.0)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
            
        # Final nudge search
        if time.time() < deadline - 5:
            nudged_pts, nudged_score = _local_nudge_search(best_points, max_iter=500, time_limit=5.0, rng=rng)
            if nudged_score > best_score:
                best_points = nudged_pts
                best_score = nudged_score
    
    # Phase 3: If we still have time, try more diverse initializations
    if best_points is None or time.time() < deadline - 25:
        if time.time() > deadline - 20:
            if best_points is not None:
                return best_points
            # Fallback: return a valid configuration
            bary = _random_valid_barycentric(rng)
            return _bary_to_cart(bary)
        
        # Try a few more Sobol-based restarts with different seeds
        for seed in [100, 200, 300]:
            if time.time() > deadline - 15:
                break
            
            sobol_pts = _sobol_points_in_triangle(N, seed=seed)
            pts, score = _try_nm_optimize(sobol_pts, beta=50.0, time_limit=8.0)
            if pts is not None and score > best_score:
                best_points = pts
                best_score = score
                
            # Apply nudge search
            if pts is not None and time.time() < deadline - 10:
                nudged_pts, nudged_score = _local_nudge_search(pts, max_iter=200, time_limit=5.0, rng=rng)
                if nudged_score > best_score:
                    best_points = nudged_pts
                    best_score = nudged_score
    
    # Final check and strict validation
    if best_points is None:
        # Fallback: return a valid configuration
        bary = _random_valid_barycentric(rng)
        best_points = _bary_to_cart(bary)
    
    # Strictly enforce containment by projecting the final output
    final_points = _project_to_triangle(best_points.copy())
    
    return final_points
