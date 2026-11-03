import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)
TRIG_AREA = SQRT3 / 4.0

V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])
CENTROID = (V1 + V2 + V3) / 3.0


def _cart_to_bary(points):
    """Convert N points from Cartesian to barycentric coordinates."""
    n = len(points)
    bary = np.zeros((n, 3))
    bary[:, 2] = points[:, 1] / (SQRT3 / 2.0)
    bary[:, 1] = points[:, 0] - 0.5 * bary[:, 2]
    bary[:, 0] = 1.0 - bary[:, 1] - bary[:, 2]
    return bary


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian."""
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * (SQRT3 / 2.0)
    return np.column_stack((x, y))


def _all_areas(points):
    """
    Compute all C(11,3)=165 triangle areas for N points in Cartesian coords.
    Returns array of 165 areas.
    """
    n = len(points)
    areas = np.empty(165)
    idx = 0
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            vj = points[j] - pi
            for k in range(j + 1, n):
                vk = points[k] - pi
                areas[idx] = 0.5 * abs(vj[0] * vk[1] - vj[1] * vk[0])
                idx += 1
    return areas


def _min_area(points):
    return np.min(_all_areas(points))


def _project_to_triangle(points):
    """Project points to be inside the triangle using barycentric clipping."""
    bary = _cart_to_bary(points)
    bary = np.maximum(bary, 0.0)
    row_sums = bary.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    bary = bary / row_sums
    return _bary_to_cart(bary)


def _generate_initial_configs(rng, n_configs):
    """Generate diverse initial configurations."""
    configs = []
    
    # Config 1: Grid-like in barycentric
    bary_grid = []
    # 3x3 grid in barycentric (interior points only)
    for a in [0.1, 0.35, 0.55]:
        for b in [0.1, 0.35, 0.55]:
            c = 1.0 - a - b
            if a >= 0.05 and b >= 0.05 and c >= 0.05:
                bary_grid.append([a, b, c])
    # Take first 11
    while len(bary_grid) < 11:
        bary_grid.append([1/3, 1/3, 1/3])
    configs.append(_bary_to_cart(np.array(bary_grid[:11])))
    
    # Config 2: Hexagonal pattern
    pts = [CENTROID.copy()]
    for k in range(6):
        angle = np.pi / 3 * k + np.pi / 6
        r = 0.25
        x = CENTROID[0] + r * np.cos(angle)
        y = CENTROID[1] + r * np.sin(angle)
        pts.append(np.array([x, y]))
    # 4 more near corners
    for frac in [0.12, 0.25]:
        for corner in [V1, V2, V3]:
            if len(pts) < 11:
                p = corner * frac + CENTROID * (1 - frac)
                pts.append(p)
    configs.append(_project_to_triangle(np.array(pts[:11])))
    
    # Config 3: Random
    for _ in range(n_configs - 2):
        pts = []
        for _ in range(N):
            for _try in range(200):
                a = rng.uniform(0.05, 0.95)
                b = rng.uniform(0.05, 0.95)
                if a + b <= 0.95:
                    pts.append([a, b, 1 - a - b])
                    break
        while len(pts) < N:
            pts.append([1/3, 1/3, 1/3])
        configs.append(_bary_to_cart(np.array(pts[:11])))
    
    return configs


def _stochastic_local_search(points, rng, time_budget, start_time):
    """
    Iterative stochastic local search: randomly perturb one point,
    accept if it improves the true minimum area.
    """
    deadline = start_time + time_budget
    current = points.copy()
    current_area = _min_area(current)
    best = current.copy()
    best_area = current_area
    
    # Adaptive perturbation scale
    scale = 0.15
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Choose a random point to perturb
        i = rng.integers(0, N)
        
        # Generate perturbation
        perturbation = rng.normal(0, scale, 2)
        new_point = current[i] + perturbation
        
        # Check if new point is inside triangle
        bary_new = _cart_to_bary(np.array([new_point]))[0]
        if bary_new[0] < -0.01 or bary_new[1] < -0.01 or bary_new[2] < -0.01:
            no_improve_count += 1
            continue
        
        # Evaluate new configuration
        trial = current.copy()
        trial[i] = new_point
        trial_area = _min_area(trial)
        
        if trial_area > current_area + 1e-12:
            current = trial
            current_area = trial_area
            no_improve_count = 0
            
            if current_area > best_area:
                best_area = current_area
                best = current.copy()
            
            # Slightly increase scale on improvement
            scale = min(scale * 1.05, 0.3)
        else:
            no_improve_count += 1
        
        # Adapt scale: decrease if stuck
        if no_improve_count >= max_no_improve:
            scale *= 0.7
            no_improve_count = 0
            if scale < 0.005:
                # Try a larger jump to escape local optimum
                scale = 0.2
                no_improve_count = 0
        
        # Occasional random restart of scale
        if no_improve_count >= max_no_improve * 3:
            scale = rng.uniform(0.05, 0.25)
            no_improve_count = 0
    
    return best, best_area


def _nelder_mead_refine(points, time_budget, start_time):
    """
    Refine using Nelder-Mead on the true objective, cycling through points.
    """
    deadline = start_time + time_budget
    current = points.copy()
    current_area = _min_area(current)
    
    for i in range(N):
        if time.time() > deadline:
            break
        
        p_i = current[i].copy()
        
        def obj_2d(x):
            trial = current.copy()
            trial[i] = x
            b = _cart_to_bary(trial)
            if np.any(b < -0.01):
                return 100.0
            return -_min_area(trial)
        
        try:
            res = minimize(
                obj_2d,
                p_i,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200}
            )
            new_p = res.x
            b_new = _cart_to_bary(np.array([new_p]))[0]
            if b_new[0] >= -1e-8 and b_new[1] >= -1e-8 and b_new[2] >= -1e-8:
                trial = current.copy()
                trial[i] = new_p
                new_area = _min_area(trial)
                if new_area > current_area + 1e-12:
                    current[i] = new_p
                    current_area = new_area
        except Exception:
            pass
    
    return current, current_area


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    rng = np.random.default_rng(12345)
    
    best_area = 0.0
    best_points = None
    
    # Generate initial configurations
    inits = _generate_initial_configs(rng, 10)
    
    # Evaluate all inits
    for init in inits:
        init = _project_to_triangle(init)
        area = _min_area(init)
        if area > best_area:
            best_area = area
            best_points = init.copy()
    
    # Sort inits by area (best first)
    init_areas = [_min_area(_project_to_triangle(i)) for i in inits]
    sorted_indices = np.argsort(init_areas)[::-1]
    
    # Run stochastic local search on top initial configs
    for idx in sorted_indices[:6]:
        if time.time() > deadline - 20:
            break
        
        remaining_time = deadline - time.time() - 15
        if remaining_time < 2:
            break
        
        init = _project_to_triangle(inits[idx])
        result, area = _stochastic_local_search(init, rng, remaining_time, start_time)
        
        if area > best_area:
            best_area = area
            best_points = result.copy()
    
    # Final Nelder-Mead refinement on the best result
    if best_points is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 3:
            refined, refined_area = _nelder_mead_refine(best_points, remaining, start_time)
            if refined_area > best_area:
                best_area = refined_area
                best_points = refined
    
    # Fallback
    if best_points is None:
        bary = np.zeros((N, 3))
        for i in range(N):
            for _try in range(1000):
                a, b = rng.uniform(0.05, 0.95, 2)
                if a + b <= 0.95:
                    bary[i] = [1 - a - b, a, b]
                    break
        best_points = _bary_to_cart(bary)
    
    return best_points
