"""Rewritten program: smooth max-min optimization via log-sum-exp proxy with multi-start L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
# Equilateral triangle vertices
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])
# Area of the triangle = sqrt(3)/4
TRI_AREA = np.sqrt(3) / 4.0

def _min_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(points)
    min_a = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a

def _smooth_min_area(points, beta=100.0):
    """
    Smooth approximation of the minimum area using log-sum-exp of reciprocals.
    We maximize min area, equivalently minimize 1/min area.
    Using: f(x) = log(sum(exp(beta / area_i))) / beta  approximates 1/min_area
    So we minimize this f(x).
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                a = 0.5 * abs(cross)
                # Avoid division by zero
                a = max(a, 1e-12)
                areas.append(1.0/a)
    areas = np.array(areas)
    # Log-sum-exp trick for numerical stability
    m = np.max(areas)
    if m > 1e10:
        # Very small areas, return a large penalty
        return 1e10
    log_sum = np.log(np.sum(np.exp(areas - m))) + m
    return log_sum / beta

def _objective_flat(flat_points, beta=100.0):
    """Objective function for L-BFGS-B: minimize smooth proxy of 1/min_area."""
    points = flat_points.reshape(-1, 2)
    # Check if points are inside the triangle (soft constraint via penalty)
    # Barycentric coordinates for equilateral triangle
    # For point (x,y), barycentric coords relative to V:
    # A point is inside if all barycentric coords are in [0,1]
    # Using: for triangle (0,0),(1,0),(0.5,sqrt(3)/2)
    # y >= 0, y <= sqrt(3)/2 * x for x in [0,0.5]... let's use a general check
    # Actually, let's use a penalty for points outside the triangle
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        # Check if inside triangle using barycentric coordinates
        # Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
        # Barycentric: 
        # d = 2 * area of triangle ABC = sqrt(3)/2
        d = np.sqrt(3)/2
        # Area of sub-triangles
        a1 = abs((points[i][0] - V[1][0])*(V[2][1] - V[1][1]) - (points[i][1] - V[1][1])*(V[2][0] - V[1][0])) / 2
        a2 = abs((V[0][0] - points[i][0])*(V[2][1] - points[i][1]) - (V[0][1] - points[i][1])*(V[2][0] - points[i][0])) / 2
        a3 = abs((points[i][0] - V[0][0])*(V[1][1] - V[0][1]) - (points[i][1] - V[0][1])*(V[1][0] - V[0][0])) / 2
        # Sum should equal TRI_AREA if inside
        total = a1 + a2 + a3
        if abs(total - TRI_AREA) > 1e-9:
            # Outside or on edge
            penalty += (total - TRI_AREA)**2 * 1000
        # Also check barycentric coords are non-negative
        # Simpler: check y >= 0, y <= sqrt(3)*x for x <= 0.5, y <= sqrt(3)*(1-x) for x >= 0.5
        if y < -1e-9:
            penalty += y**2 * 1000
        elif x <= 0.5:
            if y > np.sqrt(3)*x + 1e-9:
                penalty += (y - np.sqrt(3)*x)**2 * 1000
        else:
            if y > np.sqrt(3)*(1-x) + 1e-9:
                penalty += (y - np.sqrt(3)*(1-x))**2 * 1000
    return _smooth_min_area(points, beta) + penalty

def _random_points_in_triangle(rng):
    """Generate N random points inside the equilateral triangle."""
    points = []
    for _ in range(N):
        while True:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v <= 1:
                # Barycentric to Cartesian
                x = V[0][0] + u*(V[1][0]-V[0][0]) + v*(V[2][0]-V[0][0])
                y = V[0][1] + u*(V[1][1]-V[0][1]) + v*(V[2][1]-V[0][1])
                points.append([x, y])
                break
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Multiple random restarts with L-BFGS-B
    rng = np.random.RandomState(42)
    
    n_restarts = 0
    max_restarts = 50
    
    while n_restarts < max_restarts:
        if time.time() > deadline - 10.0:
            break
        
        # Generate initial points
        init_points = _random_points_in_triangle(rng)
        flat_init = init_points.flatten()
        
        # Run L-BFGS-B
        try:
            result = minimize(
                _objective_flat,
                flat_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < np.inf:
                opt_points = result.x.reshape(-1, 2)
                # Verify points are inside triangle
                valid = True
                for i in range(N):
                    x, y = opt_points[i]
                    if y < -1e-6:
                        valid = False
                        break
                    elif x <= 0.5:
                        if y > np.sqrt(3)*x + 1e-6:
                            valid = False
                            break
                    else:
                        if y > np.sqrt(3)*(1-x) + 1e-6:
                            valid = False
                            break
                if valid:
                    score = _min_area(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
        except Exception:
            pass
        
        n_restarts += 1
        
        if time.time() > deadline - 5.0:
            break
    
    if best_points is None:
        # Fallback: return random points
        best_points = _random_points_in_triangle(rng)
    
    return best_points
