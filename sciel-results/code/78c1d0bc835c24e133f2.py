"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by triangle area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area / AREA


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _is_inside_triangle_batch(pts):
    """Vectorized check for multiple points."""
    x = pts[:, 0]
    y = pts[:, 1]
    mask = (y >= -1e-9) & (x >= -1e-9) & (x <= 1.0 + 1e-9)
    max_y = np.where(x <= 0.5, np.sqrt(3) * x, np.sqrt(3) * (1.0 - x))
    mask &= (y <= max_y + 1e-9)
    return mask


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    # Place points in a pattern that avoids clustering
    # Use a grid-like structure within the triangle
    points = []
    # Vertices
    points.append(V0.copy())
    points.append(V1.copy())
    points.append(V2.copy())
    
    # Midpoints of edges
    points.append((V0 + V1) / 2.0)
    points.append((V1 + V2) / 2.0)
    points.append((V2 + V0) / 2.0)
    
    # Centroid
    points.append((V0 + V1 + V2) / 3.0)
    
    # Additional points - place them in a structured way
    # Use points at 1/3 and 2/3 positions
    for t in [1/3, 2/3]:
        # Along edges
        points.append(V0 * (1-t) + V1 * t)
    
    # We need exactly 11 points, we have 9 so far
    # Add 2 more points
    # Place them in the interior
    h = np.sqrt(3) / 2.0
    points.append(np.array([0.5, h * 0.5]))
    points.append(np.array([0.5, h * 0.25]))
    
    # Adjust to get exactly 11
    points = points[:N]
    return np.array(points)


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _soft_min_area(points, beta=50.0):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This is differentiable and can be used with gradient-based optimizers.
    """
    n = len(points)
    areas = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    
    areas = np.array(areas)
    # Use log-sum-exp trick for numerical stability
    # soft_min = -log(sum(exp(-beta * a))) / beta
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_terms))
    soft_min = (max_a - log_sum / beta) / AREA
    return soft_min


def _objective(params, beta=50.0):
    """
    Objective function for L-BFGS-B optimizer.
    params: flat array of shape (2*n,) representing 2D coordinates of n points.
    Returns: negative of soft min area (since L-BFGS-B minimizes).
    """
    points = params.reshape(-1, 2)
    # Check if all points are inside the triangle
    if not _is_inside_triangle_batch(points).all():
        # Penalty for being outside
        penalty = 0.0
        for p in points:
            x, y = p
            if y < -1e-9:
                penalty += abs(y)
            if x < -1e-9:
                penalty += abs(x)
            if x > 1.0 + 1e-9:
                penalty += abs(x - 1.0)
            max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
            if y > max_y + 1e-9:
                penalty += abs(y - max_y)
        return -soft_min_area(points, beta) + 100.0 * penalty
    
    return -soft_min_area(points, beta)


def _bounds():
    """Return bounds for L-BFGS-B optimizer."""
    # The triangle is bounded: x in [0, 1], y in [0, sqrt(3)/2]
    # But we need to enforce the triangular constraint separately
    # For simplicity, use loose bounds and rely on penalty
    lb = np.zeros(2 * N)
    ub = np.array([1.0, np.sqrt(3) / 2.0] * N)
    return (lb, ub)


def _optimize_with_l_bfgs_b(points, beta=50.0, time_limit=5.0):
    """
    Optimize the configuration using L-BFGS-B with soft min area objective.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    params = points.flatten().copy()
    lb, ub = _bounds()
    
    def obj_func(p):
        if time.time() - start_time > time_limit:
            return 1e10  # Signal timeout
        return _objective(p, beta)
    
    result = minimize(
        obj_func,
        params,
        method='L-BFGS-B',
        bounds=list(zip(lb, ub)),
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 2000}
    )
    
    if result.success or result.fun < 1e9:
        opt_points = result.x.reshape(-1, 2)
        opt_min = _min_triangle_area(opt_points)
        return opt_points, opt_min
    else:
        return points, _min_triangle_area(points)


def _hill_climb_from(points, max_iter=500, step_sizes=None, rng=None):
    """Hill climbing local search."""
    if rng is None:
        rng = random.Random()
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    current = points.copy()
    current_min = _min_triangle_area(current)
    best_points = current.copy()
    best_min = current_min
    
    for step_size in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(N):
                orig = current[i].copy()
                # Try random directions
                for _ in range(20):
                    angle = rng.random() * 2 * np.pi
                    dist = rng.random() * step_size
                    new_p = orig + np.array([dist * np.cos(angle), dist * np.sin(angle)])
                    if _is_inside_triangle(new_p):
                        candidate = current.copy()
                        candidate[i] = new_p
                        new_min = _min_triangle_area(candidate)
                        if new_min > current_min:
                            current[i] = new_p
                            current_min = new_min
                            improved = True
                            if new_min > best_min:
                                best_min = new_min
                                best_points = current.copy()
                            break
    
    return best_points, best_min


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy 1: Start with structured initial configuration and optimize with L-BFGS-B
    structured = _generate_structured_initial()
    opt_points, opt_min = _optimize_with_l_bfgs_b(structured, beta=50.0, time_limit=5.0)
    if opt_min > best_min:
        best_min = opt_min
        best_points = opt_points.copy()
    
    # Strategy 2: Random restarts with L-BFGS-B
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        points = _generate_random_points(rng)
        # Add a small perturbation to avoid exact duplicates or local minima
        opt_points, opt_min = _optimize_with_l_bfgs_b(points, beta=50.0, time_limit=3.0)
        if opt_min > best_min:
            best_min = opt_min
            best_points = opt_points.copy()
    
    # Strategy 3: Refine the best point found with a longer run or different beta
    if best_points is not None and time.time() < deadline - 15:
        # Try refining with a higher beta (sharper min)
        opt_points, opt_min = _optimize_with_l_bfgs_b(best_points.copy(), beta=100.0, time_limit=10.0)
        if opt_min > best_min:
            best_min = opt_min
            best_points = opt_points.copy()
    
    # Strategy 4: If we have time, try a few more random restarts with the best point as a seed
    if best_points is not None and time.time() < deadline - 10:
        for _ in range(15):
            if time.time() > deadline:
                break
            # Perturb best point
            perturbed = best_points + (rng.random(N * 2) * 0.05 - 0.025)
            perturbed = perturbed.reshape(N, 2)
            # Check if inside
            if _is_inside_triangle_batch(perturbed).all():
                opt_points, opt_min = _optimize_with_l_bfgs_b(perturbed, beta=50.0, time_limit=3.0)
                if opt_min > best_min:
                    best_min = opt_min
                    best_points = opt_points.copy()
    
    # Strategy 5: Hill climbing from best
    if best_points is not None and time.time() < deadline - 5:
        hc_points, hc_min = _hill_climb_from(best_points.copy(), max_iter=200, rng=rng)
        if hc_min > best_min:
            best_min = hc_min
            best_points = hc_points.copy()
    
    # Final check
    if best_points is None:
        best_points = _generate_structured_initial()
    
    return best_points
# EVOLVE-BLOCK-END
