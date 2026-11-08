"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 143.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _critical_triangles(points, threshold_ratio=0.15):
    """Find triangles whose area is within threshold_ratio of the minimum."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            a = 0.5 * np.abs(cross)
            for k in range(j + 1, n):
                areas.append(a[k - j - 1])
                indices.append((i, j, k))
    areas = np.array(areas)
    min_a = np.min(areas)
    if min_a < 1e-14:
        return []
    thresh = min_a * (1.0 + threshold_ratio)
    crit = [(i, j, k) for (i, j, k), a in zip(indices, areas) if a <= thresh]
    return crit


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _soft_obj_and_grad(pts_flat, beta):
    """Soft objective and its gradient. Returns (value, gradient)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10, np.zeros_like(pts_flat)
    
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
        weights = np.zeros_like(areas)
        weights[np.argmin(areas)] = 1.0
    else:
        exp_z = np.exp(z - max_z)
        log_sum = max_z + np.log(np.sum(exp_z))
        weights = exp_z / np.sum(exp_z)
    
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10, np.zeros_like(pts_flat)
    
    obj_val = hull_area / soft_min_val
    
    # Compute gradient of soft_min with respect to points
    # d(soft_min)/dx = sum_k w_k * d(area_k)/dx
    n = N
    x = points[:, 0]
    y = points[:, 1]
    
    grad_soft_min = np.zeros((n, 2))
    
    # Precompute all triangle gradients
    tri_idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                w = weights[tri_idx]
                if w > 1e-10:
                    cross = (x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i])
                    sign = np.sign(cross)
                    if sign == 0:
                        sign = 1.0
                    
                    # d(cross)/d(point_i) = (-(y[k]-y[i]), x[k]-x[i])
                    # d(cross)/d(point_j) = (y[k]-y[i], -(x[k]-x[i]))
                    # d(cross)/d(point_k) = (-(y[j]-y[i]), x[j]-x[i])
                    # d(area)/d(p) = 0.5 * sign * d(cross)/d(p)
                    
                    grad_soft_min[i, 0] += 0.5 * w * sign * (-(y[k] - y[i]))
                    grad_soft_min[i, 1] += 0.5 * w * sign * (x[k] - x[i])
                    grad_soft_min[j, 0] += 0.5 * w * sign * (y[k] - y[i])
                    grad_soft_min[j, 1] += 0.5 * w * sign * (-(x[k] - x[i]))
                    grad_soft_min[k, 0] += 0.5 * w * sign * (-(y[j] - y[i]))
                    grad_soft_min[k, 1] += 0.5 * w * sign * (x[j] - x[i])
                tri_idx += 1
    
    # d(obj)/dx = (hull_area * d(soft_min)/dx - soft_min_val * d(hull_area)/dx) / soft_min_val^2
    # Approximate d(hull_area)/dx as zero for simplicity (hull changes slowly)
    # Better: use numerical gradient for hull area
    grad_obj = hull_area * grad_soft_min / (soft_min_val ** 2)
    
    # Add a small regularization to encourage spread
    centroid = np.mean(points, axis=0)
    for i in range(n):
        dx = points[i, 0] - centroid[0]
        dy = points[i, 1] - centroid[1]
        dist = np.sqrt(dx**2 + dy**2) + 1e-10
        grad_obj[i, 0] -= 0.01 * dx / dist
        grad_obj[i, 1] -= 0.01 * dy / dist
    
    return obj_val, -grad_obj  # Negative because we minimize


def _soft_obj(pts_flat, beta=100.0):
    """Soft objective: hull_area / soft_min(triangle areas)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points for better minimum triangle area."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_hexagonal(n):
    """Hexagonal lattice inspired initialization."""
    pts = []
    pts.append([0.0, 0.0])
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _initial_cosine_mod(n):
    """Regular polygon with cosine-modulated radii."""
    angles = 2 * np.pi * np.arange(n) / n
    mod = 1.0 + 0.1 * np.cos(3 * angles)
    pts = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_grid_based(n):
    """Grid-based initialization with perturbation."""
    rng = np.random.RandomState(123)
    # Create a roughly uniform distribution
    angles = rng.uniform(0, 2*np.pi, n)
    radii = rng.uniform(0.3, 1.0, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _optimize_with_annealing(init_pts, time_budget):
    """Optimize using beta annealing with L-BFGS-B."""
    start_time = time.time()
    x0 = init_pts.flatten()
    
    best_x = x0.copy()
    best_obj = _soft_obj(x0, beta=50.0)
    
    # Annealing schedule: start with low beta (smooth), increase to high beta (sharp)
    betas = [20, 40, 80, 160, 320, 640, 1200]
    total_stages = len(betas)
    
    for si, beta in enumerate(betas):
        if time.time() - start_time > time_budget - 3:
            break
        
        # Allocate more iterations to later stages (sharper minima)
        remaining_stages = total_stages - si
        stage_budget = time_budget * (1.0 / (1.0 + 0.5 * si)) / remaining_stages
        maxiter = max(100, int(stage_budget / 0.02))
        maxfun = maxiter * 8
        
        def obj_func(x):
            return _soft_obj(x, beta=beta)
        
        def obj_grad(x):
            val, grad = _soft_obj_and_grad(x, beta)
            if grad is None:
                return val
            return val, -grad  # We're minimizing, so use negative gradient
        
        try:
            res = minimize(obj_grad, best_x, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10,
                                  'maxfun': maxfun, 'maxls': 50})
            if res.x is not None:
                cand_obj = _soft_obj(res.x, beta=beta)
                if cand_obj < best_obj:
                    best_obj = cand_obj
                    best_x = res.x
        except Exception:
            pass
    
    # Convert back to points
    cand = best_x.reshape(-1, 2)
    return cand


def _local_polish(pts, time_budget):
    """Final local polish using the exact min-triangle objective."""
    start_time = time.time()
    x0 = pts.flatten()
    
    best_x = x0.copy()
    best_score = _score(pts)
    
    # Phase 1: Nelder-Mead on the hard objective with tight tolerances
    try:
        res = minimize(_neg_score, best_x, method='Nelder-Mead',
                      options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14,
                              'adaptive': True, 'maxfev': 15000})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_x = res.x
    except Exception:
        pass
    
    # Phase 2: SLSQP with soft objective at very high beta
    if time.time() - start_time < time_budget - 5:
        try:
            def soft_high_beta(x):
                return _soft_obj(x, beta=2000.0)
            
            res = minimize(soft_high_beta, best_x, method='SLSQP',
                          options={'maxiter': 500, 'ftol': 1e-16, 'iprint': 0})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_x = res.x
        except Exception:
            pass
    
    # Phase 3: Coordinate descent on critical triangles
    if time.time() - start_time < time_budget - 3:
        try:
            current_pts = best_x.reshape(N, 2)
            for _round in range(20):
                if time.time() - start_time > time_budget - 2:
                    break
                crit = _critical_triangles(current_pts, threshold_ratio=0.05)
                if not crit:
                    break
                # For each critical triangle, try small perturbations of its vertices
                improved = False
                for (i, j, k) in crit[:10]:  # Limit to top 10
                    for pi in [i, j, k]:
                        for delta in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                            for coord in [0, 1]:
                                test_pts = current_pts.copy()
                                test_pts[pi, coord] += delta
                                s = _score(test_pts)
                                if s > best_score + 1e-12:
                                    best_score = s
                                    best_x = test_pts.flatten()
                                    current_pts = test_pts
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)


def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
        _initial_hexagonal(N),
        _initial_cosine_mod(N),
        _initial_grid_based(N),
    ]
    
    # Add perturbed versions of the regular polygon
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.06, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    # Add a "near-optimal" configuration: two concentric rings
    outer_r = 1.0
    inner_r = 0.55
    outer_angles = 2 * np.pi * np.arange(8) / 8
    inner_angles = 2 * np.pi * np.arange(6) / 6 + np.pi / 6
    outer_pts = np.column_stack([outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)])
    inner_pts = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
    two_ring = np.vstack([outer_pts, inner_pts])
    inits.append(_normalize_unit_area(two_ring))
    
    # Evaluate initial scores
    init_scores = []
    for init in inits:
        s = _score(init)
        init_scores.append(s)
    
    # Sort by initial score (best first)
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    best_pts = inits[0]
    best_score = init_scores[sorted_indices[0]]
    
    # Allocate time: ~55% for optimization, ~35% for polishing, ~10% buffer
    total_time = DEADLINE - (time.time() - start_time)
    polish_time = total_time * 0.35
    
    # Optimize from top initializations
    num_opt = 4  # Optimize from top 4 initializations
    for idx, init in enumerate(inits[:num_opt]):
        if time.time() - start_time > DEADLINE - polish_time - 4:
            break
        
        time_left = DEADLINE - polish_time - 4 - (time.time() - start_time)
        if time_left < 3:
            break
        
        # Give more time to earlier (better) initializations
        time_alloc = time_left * (1.0 / (1.0 + 0.3 * idx))
        
        try:
            cand = _optimize_with_annealing(init, time_alloc * 0.95)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass
    
    # Final polish with generous budget
    if time.time() - start_time < DEADLINE - 3:
        try:
            polish_budget = min(polish_time, DEADLINE - 4 - (time.time() - start_time))
            if polish_budget > 2:
                best_pts = _local_polish(best_pts, polish_budget)
                final_score = _score(best_pts)
                if final_score > best_score:
                    best_score = final_score
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
