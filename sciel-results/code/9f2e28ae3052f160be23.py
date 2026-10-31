import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
TOTAL_TIME_BUDGET = 140.0  # Safety margin below the 150s limit

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(x_flat, beta=50.0, deadline=None):
    """Smooth surrogate for -min(triangle_area)/hull_area using log-sum-exp."""
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    a_min_val = np.min(areas)
    if a_min_val < 1e-15:
        return 1e10
    
    ratios = areas / hull_area
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    # Log-sum-exp trick for numerical stability
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _hard_objective(x_flat, deadline=None):
    """Hard objective: -min_triangle_area / hull_area"""
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _build_template_13_0(params):
    """All 13 points on a ring. params: [radius, angle_offset]"""
    r = params[0]
    a0 = params[1]
    angles = np.linspace(a0, 2 * np.pi + a0, N, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _build_template_12_1(params):
    """12 outer + 1 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out, a_out, r_in, a_in = params[:4]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 12, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 1, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_11_2(params):
    """11 outer + 2 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out, a_out, r_in, a_in = params[:4]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 11, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 2, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_10_3(params):
    """10 outer + 3 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out, a_out, r_in, a_in = params[:4]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 10, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 3, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_9_4(params):
    """9 outer + 4 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out, a_out, r_in, a_in = params[:4]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 9, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 4, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_8_5(params):
    """8 outer + 5 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out, a_out, r_in, a_in = params[:4]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 8, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 5, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_7_6(params):
    """7 outer + 6 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out, a_out, r_in, a_in = params[:4]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 7, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 6, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _normalize_points(points):
    """Normalize points to have unit hull area and center at origin."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return points
    scale = 1.0 / np.sqrt(hull_area)
    center = np.mean(points, axis=0)
    return (points - center) * scale


def _run_optimization(initial_flat, deadline, method='Nelder-Mead', max_iter=200, beta=None, use_soft=False):
    """
    Run optimization from a given starting point.
    Returns the best points found and the score.
    """
    if time.time() > deadline:
        return None, 0.0
    
    if use_soft:
        obj_func = lambda x: _soft_min_objective(x, beta=beta, deadline=deadline)
    else:
        obj_func = lambda x: _hard_objective(x, deadline=deadline)
    
    try:
        result = minimize(
            obj_func,
            initial_flat,
            method=method,
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if time.time() > deadline:
            return None, 0.0
            
        new_points = result.x.reshape(N, 2)
        new_points = _normalize_points(new_points)
        new_score = _score(new_points)
        
        if new_score > 0:
            return new_points, new_score
        else:
            return None, 0.0
            
    except Exception:
        return None, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME_BUDGET
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()
    
    # 1. Initialize with Regular 13-gon
    init_points = _regular_polygon(N)
    init_score = _score(init_points)
    update_best(init_points, init_score)
    
    # 2. Local refinement of the regular polygon using Soft-Min objective
    #    Soft-min provides smoother gradients, helping escape shallow local optima
    if time.time() < deadline - 10:
        res_points, res_score = _run_optimization(
            init_points.flatten(), 
            deadline, 
            method='Nelder-Mead', 
            max_iter=300, 
            beta=50.0, 
            use_soft=True
        )
        update_best(res_points, res_score)
        
        # Hard polish
        if best_points is not None:
            res_points, res_score = _run_optimization(
                best_points.flatten(), 
                deadline, 
                method='Nelder-Mead', 
                max_iter=200, 
                use_soft=False
            )
            update_best(res_points, res_score)
    
    # 3. Explore structured templates with fewer outer points
    #    This is a structural change: moving points from the hull to the interior
    #    can allow for a more "dense" packing of triangles in the interior
    #    while maintaining a reasonable hull area.
    
    templates = [
        (_build_template_12_1, np.array([1.0, 0.0, 0.5, 0.0])),
        (_build_template_11_2, np.array([1.0, 0.0, 0.6, 0.0])),
        (_build_template_10_3, np.array([1.0, 0.0, 0.7, 0.0])),
        (_build_template_9_4, np.array([1.0, 0.0, 0.8, 0.0])),
        (_build_template_8_5, np.array([1.0, 0.0, 0.85, 0.0])),
        (_build_template_7_6, np.array([1.0, 0.0, 0.9, 0.0])),
    ]
    
    rng = np.random.default_rng(42)
    
    for builder, init_params in templates:
        if time.time() > deadline - 15:
            break
        
        # Try a few variations of parameters
        for _ in range(2):
            if time.time() > deadline - 15:
                break
            
            # Add small noise to break symmetry
            noise = rng.normal(0, 0.05, size=len(init_params))
            params = init_params + noise
            
            # Ensure radii are positive
            params[0] = np.abs(params[0]) + 0.5
            params[2] = np.abs(params[2]) + 0.1
            
            try:
                points = builder(params)
                points = _normalize_points(points)
                score = _score(points)
                update_best(points, score)
                
                # Optimize from this structured start
                if time.time() < deadline - 10:
                    res_points, res_score = _run_optimization(
                        points.flatten(), 
                        deadline, 
                        method='Nelder-Mead', 
                        max_iter=200, 
                        beta=50.0, 
                        use_soft=True
                    )
                    update_best(res_points, res_score)
                    
                    if best_points is not None:
                        res_points, res_score = _run_optimization(
                            best_points.flatten(), 
                            deadline, 
                            method='Nelder-Mead', 
                            max_iter=100, 
                            use_soft=False
                        )
                        update_best(res_points, res_score)
            except Exception:
                continue
    
    # 4. Final polish with random restarts if time permits
    if best_points is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            
            noise = rng.normal(0, 0.02, size=(N, 2))
            perturbed = best_points + noise
            perturbed = _normalize_points(perturbed)
            
            res_points, res_score = _run_optimization(
                perturbed.flatten(), 
                deadline, 
                method='Nelder-Mead', 
                max_iter=100, 
                use_soft=False
            )
            update_best(res_points, res_score)
    
    if best_points is None:
        return _regular_polygon(N)
        
    return best_points

# EVOLVE-BLOCK-END
