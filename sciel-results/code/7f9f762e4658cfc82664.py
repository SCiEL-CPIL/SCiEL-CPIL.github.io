"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = 3 ** 0.5
TRI_AREA = SQRT3 / 4  # Area of the equilateral triangle with side length 1

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _all_triangle_areas(points):
    """Compute all triangle areas for triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    return np.min(_all_triangle_areas(points))

def _in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    max_y = SQRT3 / 2 * min(x, 1 - x)
    if y > max_y + eps:
        return False
    return True

def _project_to_triangle(p):
    """Project a point onto the equilateral triangle."""
    x, y = p
    if y < 0:
        y = 0
    if x < 0:
        x = 0
    if x > 1:
        x = 1
    max_y = SQRT3 / 2 * min(x, 1 - x)
    if y > max_y:
        y = max_y
    return np.array([x, y])

def _feasible(params):
    """Check if all points are in the triangle."""
    points = params.reshape(N, 2)
    for i in range(N):
        if not _in_triangle(points[i], eps=1e-8):
            return False
    return True

def _penalty(params):
    """Penalty for points outside the triangle."""
    points = params.reshape(N, 2)
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        if y < 0:
            penalty += abs(y)
        if x < 0:
            penalty += abs(x)
        if x > 1:
            penalty += abs(x - 1)
        max_y = SQRT3 / 2 * min(max(x, 0), 1 - max(x, 0))
        if y > max_y:
            penalty += abs(y - max_y)
    return penalty

def _objective(params):
    """Objective function: negative of min triangle area plus penalty."""
    points = params.reshape(N, 2)
    pen = _penalty(params)
    if pen > 1e-6:
        return 1000.0 + pen
    min_area = _min_triangle_area(points)
    return -min_area

def _slsqp_refine(points, max_iter=2000):
    """Local refinement using SLSQP."""
    params0 = points.flatten()
    
    def obj(p):
        return _objective(p)
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: SQRT3/2 * p[2*i] - p[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: SQRT3/2 * (1 - p[2*i]) - p[2*i+1]})
    
    bounds = [(0, 1)] * (2 * N)
    
    try:
        result = minimize(
            obj, params0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
        )
        if result.fun < obj(params0):
            pts = result.x.reshape(N, 2)
            if _feasible(result.x):
                return pts, _min_triangle_area(pts)
    except Exception:
        pass
    
    return points, _min_triangle_area(points)

def _known_good_config():
    """A known good starting configuration based on literature."""
    # Start with a symmetric configuration
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, SQRT3/2])
    
    # Points along edges (1/3 and 2/3)
    points.append([1/3, 0.0])
    points.append([2/3, 0.0])
    points.append([1/6, SQRT3/6])
    points.append([5/6, SQRT3/6])
    points.append([1/3, SQRT3/3])
    points.append([2/3, SQRT3/3])
    
    # Center
    points.append([0.5, SQRT3/6])
    
    # One more interior point
    points.append([0.5, SQRT3/3])
    
    return np.array(points[:N])

def _random_config():
    """Generate a random configuration inside the triangle."""
    points = []
    for _ in range(N):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        b1 = 1 - u - v
        b2 = u
        b3 = v
        x = b1 * 0 + b2 * 1 + b3 * 0.5
        y = b1 * 0 + b2 * 0 + b3 * (SQRT3 / 2)
        points.append([x, y])
    return np.array(points)

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    np.random.seed(42)
    
    best_points = None
    best_val = -float('inf')
    
    # Start with a known good configuration and refine
    config = _known_good_config()
    refined, val = _slsqp_refine(config, max_iter=3000)
    if val > best_val:
        best_val = val
        best_points = refined
    
    # Try differential evolution with multiple restarts
    bounds = [(0, 1)] * (2 * N)
    
    for de_restart in range(3):
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42 + de_restart,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol'
            )
            
            if _feasible(de_result.x):
                pts = de_result.x.reshape(N, 2)
                val = _min_triangle_area(pts)
                if val > best_val:
                    best_val = val
                    best_points = pts
                
                # Refine with SLSQP
                refined, val2 = _slsqp_refine(pts, max_iter=2000)
                if val2 > best_val:
                    best_val = val2
                    best_points = refined
                    
        except Exception:
            pass
    
    # Try several random restarts with local refinement
    for r in range(5):
        config = _random_config()
        refined, val = _slsqp_refine(config, max_iter=1500)
        if val > best_val:
            best_val = val
            best_points = refined
    
    # Final projection to ensure feasibility
    if best_points is not None:
        for i in range(N):
            best_points[i] = _project_to_triangle(best_points[i])
    
    return best_points
