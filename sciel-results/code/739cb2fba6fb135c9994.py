# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 570.0  # 30s safety margin before 600s kill

_N = 13
# Precompute all triangle index combinations
_TRI_INDICES = np.array(list(itertools.combinations(range(_N), 3)))


def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    p1 = points[_TRI_INDICES[:, 0]]
    p2 = points[_TRI_INDICES[:, 1]]
    p3 = points[_TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return float(np.min(areas))


def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """
    Compute the score: A_min / A_hull.
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 1e-15:
        return 0.0
    return a_min / a_hull


def _soft_min_area(points, beta=100.0):
    """
    Differentiable soft minimum of triangle areas using log-sum-exp.
    Returns (soft_min_value, gradient_wrt_points).
    Vectorized gradient computation.
    """
    p1 = points[_TRI_INDICES[:, 0]]
    p2 = points[_TRI_INDICES[:, 1]]
    p3 = points[_TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * areas))) / beta
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_terms)
    soft_min = max_area - np.log(sum_exp) / beta
    
    weights = exp_terms / sum_exp
    signs = np.sign(cross)
    w = 0.5 * signs * weights  # shape (286,)
    
    # Gradient contributions per triangle:
    # d(soft_min)/d(points) = sum_i w_i * d(area_i)/d(points)
    # d(area_i)/dx_a = 0.5 * sign_i * (y_b - y_c)
    # d(area_i)/dy_a = 0.5 * sign_i * (x_c - x_b)
    # etc.
    
    grad = np.zeros_like(points)
    
    # For point a (index _TRI_INDICES[:,0]):
    gy_a = w * (p2[:, 1] - p3[:, 1])  # 0.5*sign*(y_b-y_c) * 2 = sign*(y_b-y_c), but w already has 0.5
    gx_a = w * (p3[:, 0] - p2[:, 0])  # 0.5*sign*(x_c-x_b)
    
    # For point b (index _TRI_INDICES[:,1]):
    gy_b = w * (p3[:, 1] - p1[:, 1])
    gx_b = w * (p1[:, 0] - p3[:, 0])
    
    # For point c (index _TRI_INDICES[:,2]):
    gy_c = w * (p1[:, 1] - p2[:, 1])
    gx_c = w * (p2[:, 0] - p1[:, 0])
    
    np.add.at(grad, (_TRI_INDICES[:, 0], 0), gx_a)
    np.add.at(grad, (_TRI_INDICES[:, 0], 1), gy_a)
    np.add.at(grad, (_TRI_INDICES[:, 1], 0), gx_b)
    np.add.at(grad, (_TRI_INDICES[:, 1], 1), gy_b)
    np.add.at(grad, (_TRI_INDICES[:, 2], 0), gx_c)
    np.add.at(grad, (_TRI_INDICES[:, 2], 1), gy_c)
    
    return soft_min, grad


def _soft_min_objective(x, beta=100.0):
    """
    Objective: minimize -soft_min_area / hull_area approximation.
    We minimize the negative of the ratio using soft-min for the numerator.
    """
    points = x.reshape(-1, 2)
    a_min_soft, _ = _soft_min_area(points, beta)
    a_hull = _hull_area(points)
    if a_hull <= 1e-15:
        return 1e10
    return -a_min_soft / a_hull


def _negative_ratio(x):
    """
    Objective: negative of exact ratio for final polishing.
    """
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s


def _optimize_with_scipy(x0, method='L-BFGS-B', maxiter=200, maxfev=10000):
    """
    Run scipy optimization and return the result.
    """
    try:
        result = minimize(
            _negative_ratio,
            x0,
            method=method,
            options={
                'maxiter': maxiter,
                'maxfev': maxfev,
                'ftol': 1e-12,
                'xtol': 1e-12
            }
        )
        return result
    except Exception:
        return None


def _optimize_soft_min(x0, beta=100.0, maxiter=300, maxfev=20000):
    """
    Optimize using soft-min surrogate with L-BFGS-B.
    """
    try:
        result = minimize(
            _soft_min_objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'maxfev': maxfev,
                'ftol': 1e-14,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        return result
    except Exception:
        return None


def _polish(x0, maxiter=100, maxfev=5000):
    """
    Final polishing with exact objective using Powell.
    """
    try:
        result = minimize(
            _negative_ratio,
            x0,
            method='Powell',
            options={
                'maxiter': maxiter,
                'maxfev': maxfev,
                'ftol': 1e-14,
                'xtol': 1e-14
            }
        )
        return result
    except Exception:
        return None


def _polish_high_precision(x0, maxiter=300, maxfev=20000):
    """
    High-precision polish with very tight tolerances.
    """
    try:
        result = minimize(
            _negative_ratio,
            x0,
            method='Powell',
            options={
                'maxiter': maxiter,
                'maxfev': maxfev,
                'ftol': 1e-16,
                'xtol': 1e-16
            }
        )
        return result
    except Exception:
        return None


def _coordinate_descent(points, n_iter=50, step_size=0.01):
    """
    Simple coordinate descent to refine the solution.
    """
    current = points.copy()
    current_score = _score(current)
    
    for _ in range(n_iter):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(_N):
            for dim in range(2):
                for direction in [-1, 1]:
                    trial = current.copy()
                    trial[i, dim] += direction * step_size
                    trial_score = _score(trial)
                    if trial_score > current_score:
                        current = trial
                        current_score = trial_score
                        improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return current


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """
    Generate points on a regular polygon.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return points


def _concentric_rings(n):
    """
    Generate points in concentric rings: center + ring of 6 + ring of 6.
    """
    points = [[0.0, 0.0]]
    # Inner ring: 6 points at radius 1
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Outer ring: 6 points at radius 2, offset by 30 degrees
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    
    points = np.array(points[:n])
    return points


def _circle_interior(n, radius=1.0, center=(0.0, 0.0)):
    """
    Generate points inside a circle (Fibonacci-like spiral for even distribution).
    """
    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    r = radius * np.sqrt((i + 0.5) / n)
    theta = i * golden_angle
    points = np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta)
    ])
    return points


def _random_points(n, seed=42):
    """
    Generate random points in a unit square.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def _generate_structured_configs(n):
    """
    Generate multiple structured initial configurations.
    """
    configs = []
    
    # Regular polygon
    configs.append(_regular_polygon(n, radius=1.0))
    
    # Concentric rings
    configs.append(_concentric_rings(n))
    
    # Fibonacci spiral
    configs.append(_circle_interior(n, radius=1.0))
    
    # Perturbed regular polygons with different radii
    for radius in [0.8, 1.0, 1.2]:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])
        configs.append(points)
    
    # Hexagonal-like: center + 6 + 6
    configs.append(_concentric_rings(n))
    
    # Grid-like arrangements
    # 4-4-4-1 pattern
    points = []
    for row in range(4):
        for col in range(4):
            points.append([col - 1.5, row - 1.5])
    # Add one more in center
    points.append([0.0, 0.0])
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    configs.append(points)
    
    # Additional structured configs
    # Offset concentric rings with different angles
    for offset in [0.0, np.pi/6, np.pi/3]:
        pts = [[0.0, 0.0]]
        for i in range(6):
            angle = i * np.pi / 3 + offset
            pts.append([np.cos(angle), np.sin(angle)])
        for i in range(6):
            angle = i * np.pi / 3 + offset + np.pi/6
            pts.append([1.8 * np.cos(angle), 1.8 * np.sin(angle)])
        pts = np.array(pts[:n])
        configs.append(pts)
    
    # Triangular grid pattern
    pts = []
    for i in range(4):
        for j in range(4):
            x = j * 0.8 + (i % 2) * 0.4
            y = i * 0.7
            pts.append([x - 1.2, y - 1.05])
    pts.append([0.0, 0.0])
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    configs.append(pts)
    
    # Regular 13-gon with slight radial perturbation
    rng = np.random.default_rng(999)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = 1.0 + 0.1 * rng.random(n)
    pts = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    configs.append(pts)
    
    return configs


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = _N
    
    # Generate diverse initializations
    initializations = _generate_structured_configs(n)
    
    # Add random initializations
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044]:
        initializations.append(_random_points(n, seed=seed))
    
    # Evaluate all initializations
    scored_inits = []
    for init in initializations:
        if len(init) != n:
            continue
        s = _score(init)
        scored_inits.append((s, init))
    
    if not scored_inits:
        rng = np.random.default_rng(42)
        best_points = rng.random((n, 2))
        return best_points
    
    scored_inits.sort(key=lambda x: -x[0])
    
    best_points = scored_inits[0][1].copy()
    best_score = scored_inits[0][0]
    
    # Multi-phase optimization on top candidates
    num_candidates = min(10, len(scored_inits))
    
    for rank in range(num_candidates):
        if time.time() > _DEADLINE:
            break
        
        s_init, init = scored_inits[rank]
        x0 = init.flatten()
        
        # Phase 1: Soft-min optimization with L-BFGS-B
        result1 = _optimize_soft_min(x0, beta=50.0, maxiter=200, maxfev=10000)
        if result1 is not None:
            x1 = result1.x
            s1 = _score(x1.reshape(-1, 2))
            if s1 > best_score:
                best_score = s1
                best_points = x1.reshape(-1, 2).copy()
            x_current = x1
        else:
            x_current = x0
        
        # Phase 2: Increase beta for sharper soft-min
        result2 = _optimize_soft_min(x_current, beta=200.0, maxiter=200, maxfev=10000)
        if result2 is not None:
            x2 = result2.x
            s2 = _score(x2.reshape(-1, 2))
            if s2 > best_score:
                best_score = s2
                best_points = x2.reshape(-1, 2).copy()
            x_current = x2
        else:
            x_current = x1 if result1 is not None else x0
        
        # Phase 3: Polish with exact objective using Powell
        result3 = _polish(x_current, maxiter=100, maxfev=5000)
        if result3 is not None:
            x3 = result3.x
            s3 = _score(x3.reshape(-1, 2))
            if s3 > best_score:
                best_score = s3
                best_points = x3.reshape(-1, 2).copy()
            x_current = x3
        
        # Phase 4: Coordinate descent refinement
        if time.time() < _DEADLINE - 10:
            refined = _coordinate_descent(x_current.reshape(-1, 2), n_iter=20, step_size=0.005)
            s_ref = _score(refined)
            if s_ref > best_score:
                best_score = s_ref
                best_points = refined.copy()
        
        # Phase 5: High-precision polish on best candidate
        if rank == 0 and time.time() < _DEADLINE - 30:
            hp_result = _polish_high_precision(x_current, maxiter=300, maxfev=20000)
            if hp_result is not None:
                x_hp = hp_result.x
                s_hp = _score(x_hp.reshape(-1, 2))
                if s_hp > best_score:
                    best_score = s_hp
                    best_points = x_hp.reshape(-1, 2).copy()
    
    # Random restarts with full optimization pipeline if time remains
    while time.time() < _DEADLINE - 60:
        seed = int(np.random.randint(0, 1000000))
        init = _random_points(n, seed=seed)
        s_init = _score(init)
        
        if s_init < best_score * 0.3:
            continue
        
        x0 = init.flatten()
        
        # Quick soft-min optimization
        result = _optimize_soft_min(x0, beta=100.0, maxiter=100, maxfev=5000)
        if result is not None:
            x_opt = result.x
            s_opt = _score(x_opt.reshape(-1, 2))
            if s_opt > best_score:
                best_score = s_opt
                best_points = x_opt.reshape(-1, 2).copy()
        
        # Polish if improved
        if s_opt > best_score * 0.95 and time.time() < _DEADLINE - 30:
            result2 = _polish(x_opt, maxiter=50, maxfev=2000)
            if result2 is not None:
                x2 = result2.x
                s2 = _score(x2.reshape(-1, 2))
                if s2 > best_score:
                    best_score = s2
                    best_points = x2.reshape(-1, 2).copy()
        
        # Also try structured perturbations of the current best
        if time.time() < _DEADLINE - 60:
            rng2 = np.random.default_rng(seed + 7)
            perturbation = 0.05 * rng2.standard_normal((n, 2))
            perturbed = best_points + perturbation
            s_pert = _score(perturbed)
            if s_pert > best_score * 0.8:
                x_pert = perturbed.flatten()
                result_p = _optimize_soft_min(x_pert, beta=100.0, maxiter=100, maxfev=5000)
                if result_p is not None:
                    xp = result_p.x
                    sp = _score(xp.reshape(-1, 2))
                    if sp > best_score:
                        best_score = sp
                        best_points = xp.reshape(-1, 2).copy()
    
    # Final validation
    if best_points is None or best_points.shape != (n, 2):
        rng = np.random.default_rng(42)
        best_points = rng.random((n, 2))
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        rng = np.random.default_rng(42)
        best_points = rng.random((n, 2))
    
    return best_points


# EVOLVE-BLOCK-END
