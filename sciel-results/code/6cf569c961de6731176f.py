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
    """
    p1 = points[_TRI_INDICES[:, 0]]
    p2 = points[_TRI_INDICES[:, 1]]
    p3 = points[_TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp: -log(sum(exp(-beta * areas))) / beta
    # For numerical stability, subtract max
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_terms)
    soft_min = max_area - np.log(sum_exp) / beta
    
    # Gradient: d(soft_min)/d(area_i) = -exp(-beta*(area_i - max)) / sum_exp
    weights = exp_terms / sum_exp
    
    # d(area_i)/d(points) for each triangle
    # area_i = 0.5 * |cross_i|
    # We need to handle the sign of cross
    signs = np.sign(cross)
    # d(area)/d(p1) = -0.5 * sign * (v2[1], -v2[0])
    # d(area)/d(p2) =  0.5 * sign * (v2[1], -v2[0]) ... wait let me recompute
    
    # cross = v1[0]*v2[1] - v1[1]*v2[0]
    # d(cross)/d(v1[0]) = v2[1], d(cross)/d(v1[1]) = -v2[0]
    # d(cross)/d(v2[0]) = -v1[1], d(cross)/d(v2[1]) = v1[0]
    # v1 = p2 - p1, v2 = p3 - p1
    # d(cross)/d(p1[0]) = -v2[1] - v1[1]... no wait
    # v1[0] = p2[0]-p1[0], v1[1] = p2[1]-p1[1]
    # v2[0] = p3[0]-p1[0], v2[1] = p3[1]-p1[1]
    # cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
    # d(cross)/d(p1[0]) = -(p3[1]-p1[1]) + (p3[0]-p1[0]) = -v2[1] + v2[0]... 
    # Let me just use the formula: area = 0.5*|cross|, and the gradient of cross w.r.t. points
    # For cross = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
    # d(cross)/dx1 = y2-y3, d(cross)/dy1 = x3-x2
    # d(cross)/dx2 = y3-y1, d(cross)/dy2 = x1-x3
    # d(cross)/dx3 = y1-y2, d(cross)/dy3 = x2-x1
    
    grad = np.zeros_like(points)
    for i in range(len(_TRI_INDICES)):
        w = 0.5 * signs[i] * weights[i]
        idx = _TRI_INDICES[i]
        a, b, c = idx
        xa, ya = points[a]
        xb, yb = points[b]
        xc, yc = points[c]
        
        grad[a, 0] += w * (yb - yc)
        grad[a, 1] += w * (xc - xb)
        grad[b, 0] += w * (yc - ya)
        grad[b, 1] += w * (xa - xc)
        grad[c, 0] += w * (ya - yb)
        grad[c, 1] += w * (xb - xa)
    
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


def _polish(x0, maxiter=200, maxfev=10000):
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
                'ftol': 1e-15,
                'xtol': 1e-15
            }
        )
        return result
    except Exception:
        return None


def _coordinate_descent(points, n_iter=100, step_size=0.005):
    """
    Coordinate descent to refine the solution with adaptive step size.
    """
    current = points.copy()
    current_score = _score(current)
    
    for _ in range(n_iter):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(_N):
            for dim in range(2):
                # Try multiple step sizes in both directions
                for step in [step_size, step_size * 0.5, step_size * 2.0]:
                    for direction in [-1, 1]:
                        trial = current.copy()
                        trial[i, dim] += direction * step
                        trial_score = _score(trial)
                        if trial_score > current_score + 1e-15:
                            current = trial
                            current_score = trial_score
                            improved = True
        if not improved:
            step_size *= 0.7
            if step_size < 1e-7:
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


def _perturbed_polygon(n, radius=1.0, noise_scale=0.1, seed=42):
    """
    Generate perturbed regular polygon points.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = radius + rng.normal(0, noise_scale, n)
    angles += rng.normal(0, noise_scale * 0.5, n)
    points = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    return points


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
    
    # Perturbed regular polygons with different radii and noise levels
    for radius in [0.8, 1.0, 1.2]:
        for noise in [0.05, 0.1, 0.15]:
            configs.append(_perturbed_polygon(n, radius=radius, noise_scale=noise, seed=int(radius*100)+int(noise*100)))
    
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
    
    # 3-4-3-3 pattern (more spread)
    points = []
    for row in range(3):
        for col in range(4):
            points.append([col - 1.5, row - 1.0])
    for row in range(3):
        for col in range(3):
            points.append([col - 1.0, row + 0.5])
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    configs.append(points)
    
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
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]:
        initializations.append(_random_points(n, seed=seed))
    
    # Evaluate all initializations
    scored_inits = []
    for init in initializations:
        if len(init) != n:
            continue
        if not np.all(np.isfinite(init)):
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
        
        # Phase 1: Soft-min optimization with L-BFGS-B (coarse)
        result1 = _optimize_soft_min(x0, beta=30.0, maxiter=200, maxfev=10000)
        if result1 is not None:
            x1 = result1.x
            s1 = _score(x1.reshape(-1, 2))
            if s1 > best_score:
                best_score = s1
                best_points = x1.reshape(-1, 2).copy()
            x_current = x1
        else:
            x_current = x0
        
        # Phase 2: Increase beta for sharper soft-min (medium)
        result2 = _optimize_soft_min(x_current, beta=100.0, maxiter=200, maxfev=10000)
        if result2 is not None:
            x2 = result2.x
            s2 = _score(x2.reshape(-1, 2))
            if s2 > best_score:
                best_score = s2
                best_points = x2.reshape(-1, 2).copy()
            x_current = x2
        else:
            x_current = x1 if result1 is not None else x0
        
        # Phase 3: Very sharp soft-min (fine)
        result3 = _optimize_soft_min(x_current, beta=500.0, maxiter=150, maxfev=8000)
        if result3 is not None:
            x3 = result3.x
            s3 = _score(x3.reshape(-1, 2))
            if s3 > best_score:
                best_score = s3
                best_points = x3.reshape(-1, 2).copy()
            x_current = x3
        else:
            x_current = x2 if result2 is not None else x_current
        
        # Phase 4: Polish with exact objective using Powell
        if time.time() < _DEADLINE - 30:
            result4 = _polish(x_current, maxiter=200, maxfev=10000)
            if result4 is not None:
                x4 = result4.x
                s4 = _score(x4.reshape(-1, 2))
                if s4 > best_score:
                    best_score = s4
                    best_points = x4.reshape(-1, 2).copy()
                x_current = x4
        
        # Phase 5: Coordinate descent refinement
        if time.time() < _DEADLINE - 15:
            refined = _coordinate_descent(x_current.reshape(-1, 2), n_iter=50, step_size=0.003)
            s_ref = _score(refined)
            if s_ref > best_score:
                best_score = s_ref
                best_points = refined.copy()
    
    # Random restarts with full optimization pipeline if time remains
    restart_count = 0
    while time.time() < _DEADLINE - 90 and restart_count < 15:
        seed = int(np.random.randint(0, 1000000))
        init = _random_points(n, seed=seed)
        s_init = _score(init)
        restart_count += 1
        
        if s_init < best_score * 0.2:
            continue
        
        x0 = init.flatten()
        
        # Quick soft-min optimization
        result = _optimize_soft_min(x0, beta=50.0, maxiter=100, maxfev=5000)
        if result is not None:
            x_opt = result.x
            s_opt = _score(x_opt.reshape(-1, 2))
            if s_opt > best_score:
                best_score = s_opt
                best_points = x_opt.reshape(-1, 2).copy()
        
        # Polish if improved
        if s_opt > best_score * 0.9 and time.time() < _DEADLINE - 40:
            result2 = _polish(x_opt, maxiter=100, maxfev=5000)
            if result2 is not None:
                x2 = result2.x
                s2 = _score(x2.reshape(-1, 2))
                if s2 > best_score:
                    best_score = s2
                    best_points = x2.reshape(-1, 2).copy()
    
    # Final high-precision polish on the best solution
    if time.time() < _DEADLINE - 20:
        final_polish = _polish(best_points.flatten(), maxiter=300, maxfev=15000)
        if final_polish is not None:
            xf = final_polish.x
            sf = _score(xf.reshape(-1, 2))
            if sf > best_score:
                best_score = sf
                best_points = xf.reshape(-1, 2).copy()
        
        # One more coordinate descent pass
        if time.time() < _DEADLINE - 10:
            refined = _coordinate_descent(best_points, n_iter=30, step_size=0.001)
            s_ref = _score(refined)
            if s_ref > best_score:
                best_score = s_ref
                best_points = refined.copy()
    
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
