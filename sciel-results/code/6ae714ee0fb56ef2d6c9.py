import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

N = 13

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
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-10:
            return 0.0
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


def _soft_min_objective(x_flat, beta=80.0):
    """Smooth surrogate for -min(triangle_area)/hull_area using log-sum-exp."""
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
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _hard_objective(x_flat):
    """Hard objective: -min_triangle_area / hull_area"""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _local_search(points, max_iter=5000, step_size=0.05, cooling=0.999, 
                  min_step=1e-8, deadline=None):
    """
    Direct coordinate perturbation local search.
    Tries random perturbations and accepts if score improves.
    Uses simulated annealing-like cooling for step size.
    """
    best_points = points.copy()
    best_score = _score(best_points)
    current_points = points.copy()
    current_score = best_score
    step = step_size
    rng = np.random.default_rng()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if step < min_step:
            break
            
        # Generate random perturbation
        perturbation = rng.normal(0, step, size=(N, 2))
        new_points = current_points + perturbation
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
            # Keep step size or increase slightly
            step = min(step * 1.05, 0.1)
        else:
            # Reject, cool down
            step *= cooling
            
    return best_points, best_score


def _build_template_13_0(params):
    """All 13 points on a regular 13-gon. params: [radius, angle_offset]"""
    r = params[0]
    a0 = params[1]
    angles = np.linspace(a0, 2 * np.pi + a0, N, endpoint=False)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def _build_template_7_6(params):
    """7 outer + 6 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out = params[0]
    a_out = params[1]
    r_in = params[2]
    a_in = params[3]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 7, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 6, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_6_7(params):
    """6 outer + 7 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out = params[0]
    a_out = params[1]
    r_in = params[2]
    a_in = params[3]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 6, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 7, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_5_4_4(params):
    """5 outer + 4 middle + 4 inner. params: [r_o, a_o, r_m, a_m, r_i, a_i]"""
    r_o, a_o, r_m, a_m, r_i, a_i = params[:6]
    o_angles = np.linspace(a_o, 2 * np.pi + a_o, 5, endpoint=False)
    m_angles = np.linspace(a_m, 2 * np.pi + a_m, 4, endpoint=False)
    i_angles = np.linspace(a_i, 2 * np.pi + a_i, 4, endpoint=False)
    o_pts = np.column_stack([r_o * np.cos(o_angles), r_o * np.sin(o_angles)])
    m_pts = np.column_stack([r_m * np.cos(m_angles), r_m * np.sin(m_angles)])
    i_pts = np.column_stack([r_i * np.cos(i_angles), r_i * np.sin(i_angles)])
    return np.vstack([o_pts, m_pts, i_pts])


def _build_template_8_5(params):
    """8 outer + 5 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out = params[0]
    a_out = params[1]
    r_in = params[2]
    a_in = params[3]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 8, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 5, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_9_4(params):
    """9 outer + 4 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out = params[0]
    a_out = params[1]
    r_in = params[2]
    a_in = params[3]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 9, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 4, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_10_3(params):
    """10 outer + 3 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out = params[0]
    a_out = params[1]
    r_in = params[2]
    a_in = params[3]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 10, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 3, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_11_2(params):
    """11 outer + 2 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out = params[0]
    a_out = params[1]
    r_in = params[2]
    a_in = params[3]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 11, endpoint=False)
    inner_angles = np.linspace(a_in, 2 * np.pi + a_in, 2, endpoint=False)
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _build_template_12_1(params):
    """12 outer + 1 inner. params: [r_out, a_out, r_in, a_in]"""
    r_out = params[0]
    a_out = params[1]
    r_in = params[2]
    a_in = params[3]
    outer_angles = np.linspace(a_out, 2 * np.pi + a_out, 12, endpoint=False)
    inner_angles = np.array([a_in])
    outer_pts = np.column_stack([r_out * np.cos(outer_angles), r_out * np.sin(outer_angles)])
    inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
    return np.vstack([outer_pts, inner_pts])


def _template_objective(params, builder):
    """Objective for template-based optimization."""
    try:
        points = builder(params)
        if len(points) != N:
            return 1e10
        score = _score(points)
        if score <= 0:
            return 1e10
        return -score
    except Exception:
        return 1e10


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin against 150s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Define all templates with their initial parameter guesses
    templates = []
    
    # Template 13_0: all on hull
    templates.append((
        _build_template_13_0,
        np.array([1.0, 0.0]),
        [[0.5, 2.0], [0.0, 2 * np.pi]]
    ))
    
    # Template 7_6
    templates.append((
        _build_template_7_6,
        np.array([1.0, 0.0, 0.5, 0.0]),
        [[0.3, 2.0], [0.0, 2 * np.pi], [0.1, 0.9], [0.0, 2 * np.pi]]
    ))
    
    # Template 6_7
    templates.append((
        _build_template_6_7,
        np.array([1.0, 0.0, 0.5, 0.0]),
        [[0.3, 2.0], [0.0, 2 * np.pi], [0.1, 0.9], [0.0, 2 * np.pi]]
    ))
    
    # Template 5_4_4
    templates.append((
        _build_template_5_4_4,
        np.array([1.0, 0.0, 0.6, 0.0, 0.3, 0.0]),
        [[0.3, 2.0], [0.0, 2 * np.pi], [0.3, 0.9], [0.0, 2 * np.pi], [0.05, 0.7], [0.0, 2 * np.pi]]
    ))
    
    # Template 8_5
    templates.append((
        _build_template_8_5,
        np.array([1.0, 0.0, 0.5, 0.0]),
        [[0.3, 2.0], [0.0, 2 * np.pi], [0.1, 0.9], [0.0, 2 * np.pi]]
    ))
    
    # Template 9_4
    templates.append((
        _build_template_9_4,
        np.array([1.0, 0.0, 0.5, 0.0]),
        [[0.3, 2.0], [0.0, 2 * np.pi], [0.1, 0.9], [0.0, 2 * np.pi]]
    ))
    
    # Template 10_3
    templates.append((
        _build_template_10_3,
        np.array([1.0, 0.0, 0.5, 0.0]),
        [[0.3, 2.0], [0.0, 2 * np.pi], [0.1, 0.9], [0.0, 2 * np.pi]]
    ))
    
    # Template 11_2
    templates.append((
        _build_template_11_2,
        np.array([1.0, 0.0, 0.5, 0.0]),
        [[0.3, 2.0], [0.0, 2 * np.pi], [0.1, 0.9], [0.0, 2 * np.pi]]
    ))
    
    # Template 12_1
    templates.append((
        _build_template_12_1,
        np.array([1.0, 0.0, 0.3, 0.0]),
        [[0.3, 2.0], [0.0, 2 * np.pi], [0.0, 0.8], [0.0, 2 * np.pi]]
    ))
    
    # Phase 1: Evaluate initial configurations from templates
    for builder, init_params, bounds_list in templates:
        points = builder(init_params)
        _update_best(points)
    
    # Also add the regular polygon as a baseline
    _update_best(_regular_polygon(N, 1.0))
    
    # Phase 2: Optimize each template's parameters with Nelder-Mead
    for builder, init_params, bounds_list in templates:
        if time.time() > deadline - 40:
            break
        
        try:
            result = minimize(
                _template_objective,
                init_params,
                args=(builder,),
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            opt_points = builder(result.x)
            _update_best(opt_points)
            
            # Try a few random restarts for this template
            for restart in range(3):
                if time.time() > deadline - 35:
                    break
                rng = np.random.default_rng(restart * 1000 + hash(builder) % 1000)
                rand_params = np.array([
                    rng.uniform(b[0], b[1]) for b in bounds_list
                ])
                try:
                    res2 = minimize(
                        _template_objective,
                        rand_params,
                        args=(builder,),
                        method='Nelder-Mead',
                        options={'maxiter': 80, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    opt_pts2 = builder(res2.x)
                    _update_best(opt_pts2)
                except Exception:
                    pass
        except Exception:
            pass
    
    # Phase 3: Local search from best candidates
    if best_points is not None and time.time() < deadline - 20:
        # Multiple local search runs with different parameters
        for ls_iter in range(5):
            if time.time() > deadline - 10:
                break
            
            # Start from best point with small random perturbation
            rng = np.random.default_rng(ls_iter * 12345)
            perturbed = best_points + rng.normal(0, 0.01, size=(N, 2))
            
            ls_points, ls_score = _local_search(
                perturbed, 
                max_iter=3000, 
                step_size=0.03, 
                cooling=0.9995,
                deadline=deadline - 5
            )
            _update_best(ls_points)
    
    # Phase 4: More random restarts with local search if time permits
    while time.time() < deadline - 8:
        rng = np.random.default_rng()
        
        # Choose a random template
        idx = rng.integers(0, len(templates))
        builder, init_params, bounds_list = templates[idx]
        
        # Random parameters within bounds
        rand_params = np.array([
            rng.uniform(b[0], b[1]) for b in bounds_list
        ])
        
        try:
            result = minimize(
                _template_objective,
                rand_params,
                args=(builder,),
                method='Nelder-Mead',
                options={'maxiter': 60, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            opt_points = builder(result.x)
            _update_best(opt_points)
            
            # Quick local search polish if time allows
            if best_score > 0 and time.time() < deadline - 5:
                ls_points, ls_score = _local_search(
                    opt_points,
                    max_iter=1500,
                    step_size=0.02,
                    cooling=0.999,
                    deadline=deadline - 2
                )
                _update_best(ls_points)
        except Exception:
            pass
    
    # Final check
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
