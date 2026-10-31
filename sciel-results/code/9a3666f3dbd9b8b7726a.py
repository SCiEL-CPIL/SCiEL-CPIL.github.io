"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
# Vertices of the equilateral triangle with area 1 (side length sqrt(2/sqrt(3)))
# Actually, the task says "unit-area triangle" but the interface says equilateral (0,0),(1,0),(1/2,sqrt(3)/2)
# That triangle has area sqrt(3)/4. The task says "divides by its area" so we work in that coordinate system.
# The evaluator measures raw value. Let's assume the objective is the minimum area of any triangle formed by 3 points,
# and we want to maximize it. The area is computed in the given coordinate system.

SQRT3 = np.sqrt(3)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2])
TRI_AREA = SQRT3 / 4  # area of the containing triangle


def triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by points p1, p2, p3."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the given points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def points_in_triangle(points):
    """Check if all points are inside or on the boundary of the triangle."""
    for p in points:
        # Barycentric coordinate check
        # p = a*V0 + b*V1 + c*V2 where a+b+c=1, a,b,c >= 0
        # Solve for b, c given p = V0 + b*(V1-V0) + c*(V2-V0)
        # V1-V0 = (1, 0), V2-V0 = (0.5, sqrt(3)/2)
        # p_x = b + 0.5*c
        # p_y = sqrt(3)/2 * c
        c = p[1] / (SQRT3 / 2)
        b = p[0] - 0.5 * c
        a = 1 - b - c
        if a < -1e-9 or b < -1e-9 or c < -1e-9:
            return False
    return True


def objective_flat(x):
    """Objective function for optimizer: minimize negative of min triangle area."""
    points = x.reshape((-1, 2))
    # Penalty if any point is outside the triangle
    for p in points:
        c = p[1] / (SQRT3 / 2)
        b = p[0] - 0.5 * c
        a = 1 - b - c
        if a < 0 or b < 0 or c < 0:
            return 1000.0  # large penalty
    min_a = min_triangle_area(points)
    return -min_a


def construct():
    start_time = time.time()
    deadline = start_time + 130  # leave 10s safety margin
    
    best_points = None
    best_score = -float('inf')
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy: Use differential evolution for global search, then local refinement
    # The objective is to maximize min triangle area, so we minimize -min_area
    
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, SQRT3 / 2)] * N
    
    # Initial random restarts
    rng = np.random.RandomState(42)
    
    # Phase 1: Differential Evolution with multiple restarts
    # Use a smaller population to fit in time
    n_restarts = 15
    maxiter_de = 50
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial population
        x0 = rng.uniform(0.05, 0.95, size=(N, 2))
        # Project into triangle
        for i in range(N):
            u, v = x0[i]
            if u + v > 1:
                s = (u + v)
                x0[i] = u / s, v / s
            # Ensure inside triangle
            c = x0[i][1] / (SQRT3 / 2)
            b = x0[i][0] - 0.5 * c
            a = 1 - b - c
            if a < 0 or b < 0 or c < 0:
                # Re-sample
                u, v = rng.uniform(0, 1), rng.uniform(0, 1)
                if u + v < 1:
                    x0[i] = u + 0.5 * v, (SQRT3 / 2) * v
                else:
                    x0[i] = 0.5, SQRT3 / 6
        
        # Run differential evolution
        try:
            result = differential_evolution(
                objective_flat, bounds,
                seed=r, maxiter=maxiter_de, popsize=15,
                tol=1e-6, init='latinhypercube'
            )
            if result.success:
                points = result.x.reshape((-1, 2))
                if points_in_triangle(points):
                    score = min_triangle_area(points)
                    update_best(points, score)
        except Exception:
            pass
    
    # Phase 2: Local refinement using L-BFGS-B from best points and random starts
    if best_points is not None:
        # Refine the best point found
        try:
            res = minimize(
                objective_flat, best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            points = res.x.reshape((-1, 2))
            if points_in_triangle(points):
                score = min_triangle_area(points)
                update_best(points, score)
        except Exception:
            pass
    
    # More random starts for local refinement
    n_local_restarts = 50
    for i in range(n_local_restarts):
        if time.time() > deadline:
            break
        
        # Generate random points inside triangle
        points = np.zeros((N, 2))
        for j in range(N):
            while True:
                u, v = rng.uniform(0, 1), rng.uniform(0, 1)
                if u + v < 1:
                    break
            points[j] = [u + 0.5 * v, (SQRT3 / 2) * v]
        
        try:
            res = minimize(
                objective_flat, points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res.success:
                points_opt = res.x.reshape((-1, 2))
                if points_in_triangle(points_opt):
                    score = min_triangle_area(points_opt)
                    update_best(points_opt, score)
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polishing
    if best_points is not None:
        try:
            res = minimize(
                objective_flat, best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            points = res.x.reshape((-1, 2))
            if points_in_triangle(points):
                score = min_triangle_area(points)
                update_best(points, score)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return the initial random points
        points = []
        rng2 = np.random.RandomState(7)
        for _ in range(N):
            while True:
                u, v = rng2.uniform(0, 1), rng2.uniform(0, 1)
                if u + v < 0.98:
                    break
            points.append([u + 0.5 * v, (SQRT3 / 2) * v])
        best_points = np.array(points)
    
    return best_points
