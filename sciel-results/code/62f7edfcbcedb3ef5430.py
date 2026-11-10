# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull


def _triangle_areas_all(pts):
    """Compute all 286 triangle areas for 13 points, vectorized."""
    n = len(pts)
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    x1, y1 = pts[i_idx, 0], pts[i_idx, 1]
    x2, y2 = pts[j_idx, 0], pts[j_idx, 1]
    x3, y3 = pts[k_idx, 0], pts[k_idx, 1]
    # area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| / 2
    cross = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    areas = np.abs(cross) / 2.0
    return areas, i_idx, j_idx, k_idx, cross


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Compute A_min / A_hull."""
    if len(pts) < 3:
        return 0.0
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(min_area) or not np.isfinite(a_hull):
        return 0.0
    return min_area / a_hull


def _soft_min_area_and_grad(x, beta):
    """
    Compute soft-min of triangle areas and its gradient w.r.t. x (26-dim).
    x = [x0, y0, x1, y1, ..., x12, y12]
    Returns (soft_min_value, gradient_array).
    """
    n = 13
    pts = np.column_stack([x[0:13], x[13:26]])
    
    # Precompute all triangle indices
    i_list, j_list, k_list = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_list.append(i)
                j_list.append(j)
                k_list.append(k)
    i_arr = np.array(i_list)
    j_arr = np.array(j_list)
    k_arr = np.array(k_list)
    m = len(i_arr)  # 286
    
    x1, y1 = pts[i_arr, 0], pts[i_arr, 1]
    x2, y2 = pts[j_arr, 0], pts[j_arr, 1]
    x3, y3 = pts[k_arr, 0], pts[k_arr, 1]
    
    # cross = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
    cross = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    areas = np.abs(cross) / 2.0
    areas = np.maximum(areas, 1e-15)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_vals = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_vals)
    log_sum = max_val + np.log(sum_exp)
    soft_min = -log_sum / beta
    
    # Gradient of soft_min w.r.t. areas:
    # d(soft_min)/d(areas_t) = exp(-beta * areas_t) / sum(exp(-beta * areas))
    # = exp_vals[t] / sum_exp
    weights = exp_vals / sum_exp  # sum to 1
    
    # d(areas_t)/d(x_p) where areas_t = |cross_t| / 2
    # If cross_t > 0: d(areas_t)/d(cross_t) = 1/2
    # If cross_t < 0: d(areas_t)/d(cross_t) = -1/2
    # So d(areas_t)/d(cross_t) = sign(cross_t) / 2
    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0
    darea_dcross = sign_cross / 2.0
    
    # d(cross_t)/d(x_p):
    # cross = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
    # For point p:
    #   if p == i: d(cross)/d(x_i) = -(y2-y3) = y3-y2, d(cross)/d(y_i) = x2-x3
    #   if p == j: d(cross)/d(x_j) = -(y3-y1) = y1-y3, d(cross)/d(y_j) = x3-x1
    #   if p == k: d(cross)/d(x_k) = -(y1-y2) = y2-y1, d(cross)/d(y_k) = x1-x2
    
    grad = np.zeros(26)
    
    # For each triangle t, accumulate contribution to gradient
    # d(soft_min)/d(x_p) = sum_t weights[t] * darea_dcross[t] * d(cross_t)/d(x_p)
    
    # Process each triangle
    for t in range(m):
        w = weights[t] * darea_dcross[t]
        i, j, k = i_arr[t], j_arr[t], k_arr[t]
        xi, yi = pts[i]
        xj, yj = pts[j]
        xk, yk = pts[k]
        
        # Contribution to point i
        grad[2*i] += w * (yk - yj)   # d(cross)/d(x_i) = y3-y2
        grad[2*i+1] += w * (xj - xk) # d(cross)/d(y_i) = x2-x3
        
        # Contribution to point j
        grad[2*j] += w * (yi - yk)   # d(cross)/d(x_j) = y1-y3
        grad[2*j+1] += w * (xk - xi) # d(cross)/d(y_j) = x3-x1
        
        # Contribution to point k
        grad[2*k] += w * (yj - yi)   # d(cross)/d(x_k) = y2-y1
        grad[2*k+1] += w * (xi - xj) # d(cross)/d(y_k) = x1-x2
    
    # Now divide by hull area for the ratio objective
    # We want to maximize soft_min / hull_area
    # So minimize -soft_min / hull_area
    # But hull_area depends on points too, which complicates gradient.
    # For simplicity, treat hull_area as constant in gradient (approximation)
    # and refine with exact evaluation.
    
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_hull):
        a_hull = 1.0
    
    # Objective: -soft_min / a_hull
    # Gradient: -grad_soft_min / a_hull (ignoring derivative of a_hull)
    obj_val = -soft_min / a_hull
    grad_obj = -grad / a_hull
    
    return obj_val, grad_obj


def _objective_analytical(x, beta):
    """Objective value only (for evaluation)."""
    val, _ = _soft_min_area_and_grad(x, beta)
    return val


def _objective_and_grad(x, beta):
    """Objective value and gradient."""
    return _soft_min_area_and_grad(x, beta)


def _exact_objective(x):
    """Exact objective: -A_min / A_hull."""
    pts = np.column_stack([x[0:13], x[13:26]])
    s = _score(pts)
    return -s if s > 0 else 1e10


def _regular_polygon(n=13):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _generate_initializations(n=13, rng=None):
    """Generate diverse initializations."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    inits = []
    
    # 1. Regular polygon
    inits.append(_regular_polygon(n))
    
    # 2. Concentric rings
    for r_inner, n_inner, n_outer in [(0.4, 4, 9), (0.5, 5, 8), (0.5, 6, 7), (0.3, 3, 10), (0.6, 7, 6)]:
        inner_angles = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
        outer_angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
        inner_pts = np.column_stack([r_inner*np.cos(inner_angles), r_inner*np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inits.append(np.vstack([inner_pts, outer_pts]))
    
    # 3. Grid with perturbation
    for scale in [0.3, 0.5, 0.7]:
        grid = []
        for i in range(4):
            for j in range(4):
                grid.append([scale*(i-1.5) + rng.normal(0, 0.05), scale*(j-1.5) + rng.normal(0, 0.05)])
        inits.append(np.array(grid[:n]))
    
    # 4. Random points in disk
    for _ in range(8):
        angles = rng.uniform(0, 2*np.pi, n)
        radii = rng.uniform(0.3, 1.0, n)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        inits.append(pts)
    
    # 5. Perturbed regular polygon
    for pert_scale in [0.05, 0.1, 0.2, 0.3]:
        base = _regular_polygon(n)
        pts = base + rng.normal(0, pert_scale, (n, 2))
        inits.append(pts)
    
    # 6. Two clusters
    for _ in range(4):
        c1 = rng.normal(0, 0.3, (7, 2))
        c2 = rng.normal(0, 0.3, (6, 2)) + np.array([1.5, 0])
        inits.append(np.vstack([c1, c2]))
    
    # 7. Hexagonal lattice subset
    for _ in range(3):
        pts = []
        for i in range(4):
            for j in range(4):
                x = i * 0.5 + (j % 2) * 0.25
                y = j * 0.433
                pts.append([x, y])
        pts = np.array(pts[:n])
        pts = pts - pts.mean(axis=0)
        pts *= rng.uniform(0.8, 1.2)
        inits.append(pts)
    
    return inits


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    time_limit = 620  # Safety margin
    
    n = 13
    best_pts = _regular_polygon(n)
    best_score = _score(best_pts)
    
    def elapsed():
        return time.time() - start_time
    
    def update_best(pts):
        nonlocal best_pts, best_score
        if not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Phase 1: Differential evolution for global search
    if elapsed() < time_limit - 120:
        bounds = [(-5, 5)] * 26
        rng = np.random.default_rng(123)
        
        def de_objective(x):
            pts = np.column_stack([x[0:13], x[13:26]])
            s = _score(pts)
            return -s if s > 0 else 1e10
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=123,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol',
                workers=1
            )
            if de_result.fun < -best_score:
                de_pts = np.column_stack([de_result.x[0:13], de_result.x[13:26]])
                update_best(de_pts)
        except Exception:
            pass
    
    # Phase 2: Local optimization from multiple initializations using analytical gradient
    beta_schedule = [20, 50, 100, 200, 500, 1000, 2000]
    
    inits = _generate_initializations(n, np.random.default_rng(42))
    
    for init_idx, init_pts in enumerate(inits):
        if elapsed() > time_limit - 60:
            break
        
        x0 = init_pts.flatten()
        current_x = x0.copy()
        
        for beta in beta_schedule:
            if elapsed() > time_limit - 30:
                break
            
            # L-BFGS-B with analytical gradient
            try:
                def obj_func(x):
                    return _objective_analytical(x, beta)
                
                def obj_grad(x):
                    _, g = _soft_min_area_and_grad(x, beta)
                    return g
                
                result = minimize(
                    obj_func,
                    current_x,
                    method='L-BFGS-B',
                    jac=obj_grad,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
                if result.fun < _objective_analytical(current_x, beta) - 1e-12:
                    current_x = result.x
            except Exception:
                pass
            
            # Nelder-Mead polish with exact objective
            if elapsed() < time_limit - 20:
                try:
                    nm_result = minimize(
                        _exact_objective,
                        current_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 2000,
                            'xatol': 1e-12,
                            'fatol': 1e-14
                        }
                    )
                    if nm_result.fun < -best_score:
                        current_x = nm_result.x
                except Exception:
                    pass
        
        final_pts = np.column_stack([current_x[0:13], current_x[13:26]])
        update_best(final_pts)
    
    # Phase 3: Fine-tune best solution with high-beta soft-min
    if elapsed() < time_limit - 30 and best_score > 0:
        x_final = best_pts.flatten()
        
        for beta in [500, 1000, 2000, 5000]:
            if elapsed() > time_limit - 15:
                break
            try:
                def obj_func(x):
                    return _objective_analytical(x, beta)
                
                def obj_grad(x):
                    _, g = _soft_min_area_and_grad(x, beta)
                    return g
                
                result = minimize(
                    obj_func,
                    x_final,
                    method='L-BFGS-B',
                    jac=obj_grad,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'maxls': 100
                    }
                )
                if result.fun < _objective_analytical(x_final, beta) - 1e-14:
                    x_final = result.x
            except Exception:
                pass
        
        # Final Nelder-Mead with exact objective
        if elapsed() < time_limit - 5:
            try:
                nm_final = minimize(
                    _exact_objective,
                    x_final,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-14,
                        'fatol': 1e-16
                    }
                )
                if nm_final.fun < -best_score:
                    best_pts = np.column_stack([nm_final.x[0:13], nm_final.x[13:26]])
                    best_score = -nm_final.fun
            except Exception:
                pass
    
    # Validate
    if best_score <= 0 or not np.all(np.isfinite(best_pts)):
        return _regular_polygon(n)
    
    return best_pts


# EVOLVE-BLOCK-END
