"""Rewritten program: log-barrier convexity + soft-min gradient optimization."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _soft_min_triangle_area(points, temperature=0.05):
    """
    Compute a smooth approximation of the minimum triangle area using a
    negative log-sum-exp of triangle areas.
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -T * log(sum(exp(-area/T)))
    # For numerical stability
    min_val = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - min_val) / temperature)))
    soft_min = min_val - temperature * log_sum
    return soft_min


def _convexity_violation(points):
    """
    Compute a measure of non-convexity for the ordered points.
    If points are in convex position and ordered, all cross products
    of consecutive edges should have the same sign.
    """
    n = len(points)
    violations = 0.0
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        edge1 = p1 - p0
        edge2 = p2 - p1
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        # We want all crosses to be positive (CCW) or all negative (CW)
        # Penalize if any cross is small or negative
        if cross < 0.01:
            violations += (0.01 - cross) ** 2
    return violations


def _objective_cartesian(x, temperature=0.05):
    """
    Objective for minimization: negative soft-min triangle area / hull area
    plus convexity penalty.
    """
    points = x.reshape(-1, 2)
    
    # Check convexity
    cvx_viol = _convexity_violation(points)
    if cvx_viol > 1.0:
        return 1000.0 + cvx_viol
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 1000.0
    
    soft_min_area = _soft_min_triangle_area(points, temperature)
    score = soft_min_area / hull_area
    
    # Penalize convexity violations softly
    penalty = cvx_viol * 10.0
    
    return -score + penalty


def _objective_angular(angles, temperature=0.05):
    """
    Objective for angular parameterization: points on a circle with
    varying radii. This ensures convexity by construction if radii are
    positive and angles are sorted.
    """
    n = len(angles)
    # angles: [r0, a0, r1, a1, ..., r13, a13]
    radii = angles[0::2]
    thetas = angles[1::2]
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    # Sort by angle to ensure convex ordering
    order = np.argsort(thetas)
    radii = radii[order]
    thetas = thetas[order]
    
    points = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 1000.0
    
    soft_min_area = _soft_min_triangle_area(points, temperature)
    score = soft_min_area / hull_area
    
    return -score


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Initial good starting point: regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Normalize to unit area
    init_hull_area = _convex_hull_area(init_points)
    if init_hull_area > 1e-12:
        init_points = init_points / np.sqrt(init_hull_area)

    best_points = init_points.copy()
    best_score = _score(init_points)

    # Try multiple temperature values for soft-min
    temperatures = [0.02, 0.05, 0.1, 0.2]
    
    for temp in temperatures:
        if time.time() > deadline - 10.0:
            break
        
        # Cartesian optimization
        try:
            x0 = best_points.ravel()
            result = minimize(
                _objective_cartesian,
                x0,
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success:
                candidate_points = result.x.reshape(-1, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
        
        # Angular optimization
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Extract initial radii and angles from best_points
            center = np.mean(best_points, axis=0)
            centered = best_points - center
            init_radii = np.sqrt(np.sum(centered**2, axis=1))
            init_thetas = np.arctan2(centered[:, 1], centered[:, 0])
            
            # Sort by angle
            order = np.argsort(init_thetas)
            init_radii = init_radii[order]
            init_thetas = init_thetas[order]
            
            # Interleave radii and angles
            x0_angular = np.zeros(2 * N)
            x0_angular[0::2] = init_radii
            x0_angular[1::2] = init_thetas
            
            result = minimize(
                _objective_angular,
                x0_angular,
                args=(temp,),
                method='L-BFGS-B',
                bounds=[(0.1, 5.0), (-np.pi, np.pi)] * N,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success:
                radii = np.maximum(result.x[0::2], 0.1)
                thetas = result.x[1::2]
                order = np.argsort(thetas)
                radii = radii[order]
                thetas = thetas[order]
                candidate_points = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Final refinement with Nelder-Mead on the best result
    if time.time() < deadline - 5.0:
        try:
            x0 = best_points.ravel()
            result = minimize(
                _objective_cartesian,
                x0,
                args=(0.02,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.success:
                candidate_points = result.x.reshape(-1, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
