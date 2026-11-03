"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools
import sys

N = 11
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2  # height of equilateral triangle with side 1


def barycentric_coords(p):
    """Compute barycentric coordinates of point p with respect to triangle ABC."""
    v0 = B - A
    v1 = C - A
    v2 = p - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def inside_triangle(p):
    """Check if point p is inside or on the boundary of triangle ABC."""
    u, v, w = barycentric_coords(p)
    eps = 1e-9
    return u >= -eps and v >= -eps and w >= -eps


def clip_to_triangle(p):
    """Clip point p to the triangle ABC."""
    u, v, w = barycentric_coords(p)
    eps = 1e-9
    if u < -eps or v < -eps or w < -eps:
        # Find the closest point on the triangle boundary
        # This is a simplification; for robustness we project onto edges
        # For now, just return the point as-is and let the optimizer handle it
        pass
    # Return the point; the penalty function will handle constraints
    return p


def triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by three points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three of the points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def penalty(points):
    """Penalty for points outside the triangle."""
    pen = 0.0
    for p in points:
        u, v, w = barycentric_coords(p)
        if u < 0:
            pen += (-u) ** 2
        if v < 0:
            pen += (-v) ** 2
        if w < 0:
            pen += (-w) ** 2
    return pen


def objective(x):
    """Objective function to minimize (negative of min area plus penalty)."""
    points = x.reshape(-1, 2)
    pen = penalty(points)
    if pen > 1e-6:
        return 1.0 + pen  # Large penalty
    min_area = min_triangle_area(points)
    return -min_area


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin

    # Initial random points
    np.random.seed(42)
    best_points = None
    best_score = -float('inf')

    # Generate initial candidate using random sampling with rejection
    points_list = []
    while len(points_list) < N:
        u, v = np.random.random(2)
        if u + v < 1.0 and u > 0.02 and v > 0.02:
            p = np.array([u + 0.5 * v, H * v])
            points_list.append(p)
    points = np.array(points_list)
    score = min_triangle_area(points)
    if score > best_score:
        best_score = score
        best_points = points.copy()

    # Use differential evolution for global optimization
    # Flatten the points: 2*N variables
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    # Add a small margin for the triangle constraint
    # The barycentric constraint is: for point (x,y), we need
    # y >= 0, y <= H*(1-x) for x in [0, 0.5], and y <= H*x for x in [0.5, 1]
    # More precisely: y >= 0, y <= H, and 2*y/sqrt(3) <= x + (1-x)*1... 
    # Let's use the barycentric constraint directly in the penalty

    try:
        result = differential_evolution(
            objective, bounds, maxiter=50, popsize=15, tol=0.01,
            seed=42, polish=False, init='sobol'
        )
        elapsed = time.time() - start_time
        if elapsed < deadline:
            points_de = result.x.reshape(-1, 2)
            # Check if valid
            if penalty(points_de) < 1e-6:
                score_de = min_triangle_area(points_de)
                if score_de > best_score:
                    best_score = score_de
                    best_points = points_de.copy()
    except Exception as e:
        pass

    # Local optimization with L-BFGS-B
    if best_points is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            try:
                result_local = minimize(
                    objective, best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                points_local = result_local.x.reshape(-1, 2)
                if penalty(points_local) < 1e-6:
                    score_local = min_triangle_area(points_local)
                    if score_local > best_score:
                        best_score = score_local
                        best_points = points_local.copy()
            except Exception as e:
                pass

    # Multiple restarts with different seeds
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        for seed in range(10):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            np.random.seed(seed)
            points_list = []
            while len(points_list) < N:
                u, v = np.random.random(2)
                if u + v < 1.0 and u > 0.02 and v > 0.02:
                    p = np.array([u + 0.5 * v, H * v])
                    points_list.append(p)
            points = np.array(points_list)
            
            try:
                result_de = differential_evolution(
                    objective, bounds, maxiter=30, popsize=10, tol=0.01,
                    seed=seed, polish=False, init='sobol'
                )
                points_de = result_de.x.reshape(-1, 2)
                if penalty(points_de) < 1e-6:
                    score_de = min_triangle_area(points_de)
                    if score_de > best_score:
                        best_score = score_de
                        best_points = points_de.copy()
                
                # Local polish
                elapsed = time.time() - start_time
                if elapsed < deadline - 5:
                    result_local = minimize(
                        objective, points_de.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    points_local = result_local.x.reshape(-1, 2)
                    if penalty(points_local) < 1e-6:
                        score_local = min_triangle_area(points_local)
                        if score_local > best_score:
                            best_score = score_local
                            best_points = points_local.copy()
            except Exception as e:
                pass

    # Final check and return
    if best_points is None:
        # Fallback to initial random points
        points_list = []
        while len(points_list) < N:
            u, v = np.random.random(2)
            if u + v < 1.0 and u > 0.02 and v > 0.02:
                p = np.array([u + 0.5 * v, H * v])
                points_list.append(p)
        best_points = np.array(points_list)

    # Ensure all points are inside the triangle
    final_points = []
    for p in best_points:
        u, v, w = barycentric_coords(p)
        if u < 0 or v < 0 or w < 0:
            # Project to nearest edge
            # Simple projection: clip barycentric coordinates
            u = max(0, u)
            v = max(0, v)
            w = max(0, w)
            s = u + v + w
            u, v, w = u / s, v / s, w / s
            p = u * A + v * B + w * C
        final_points.append(p)
    best_points = np.array(final_points)

    return best_points
