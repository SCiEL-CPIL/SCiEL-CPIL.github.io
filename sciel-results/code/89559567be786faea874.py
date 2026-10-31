"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

SQRT3 = np.sqrt(3)
HALF_SQRT3 = SQRT3 / 2.0

def get_min_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    min_area = np.inf
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                if area < min_area:
                    min_area = area
    return min_area

def is_inside_triangle(points):
    """
    Check if all points are inside or on the boundary of the triangle
    (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    for p in points:
        x, y = p
        if y < -1e-9:
            return False
        if x < -1e-9 or x > 1 + 1e-9:
            return False
        if y > SQRT3 * x + 1e-9:
            return False
        if y > SQRT3 * (1 - x) + 1e-9:
            return False
    return True

def objective_DE(x):
    """
    Objective for Differential Evolution.
    Returns cost (to be minimized).
    """
    points = x.reshape(-1, 2)
    if not is_inside_triangle(points):
        return 1e6 # Large penalty
    area = get_min_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s buffer for safety
    
    # 1. Global Search with Differential Evolution
    bounds = [(0, 1), (0, HALF_SQRT3)] * N
    
    try:
        res_de = differential_evolution(
            objective_DE, 
            bounds, 
            seed=7, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5,
            polish=False
        )
    except Exception as e:
        # Fallback to random if DE fails
        points = np.random.rand(N, 2)
        points[:, 0] *= 1.0
        points[:, 1] *= HALF_SQRT3
        return points

    best_points_de = res_de.x.reshape(-1, 2)
    best_val_de = -res_de.fun
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_points_de

    # 2. Local Refinement with SLSQP
    def slsqp_objective(v):
        t = v[-1]
        return -t

    constraints = []
    
    for i in range(N):
        xi = i * 2
        yi = i * 2 + 1
        
        constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi: v[yi]})
        constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, yi=yi: SQRT3 * v[xi] - v[yi]})
        constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, yi=yi: SQRT3 * (1 - v[xi]) - v[yi]})

    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                xi, yi = i * 2, i * 2 + 1
                xj, yj = j * 2, j * 2 + 1
                xk, yk = k * 2, k * 2 + 1
                
                def area_constraint(v, xi=xi, yi=yi, xj=xj, yj=yj, xk=xk, yk=yk):
                    x1, y1 = v[xi], v[yi]
                    x2, y2 = v[xj], v[yj]
                    x3, y3 = v[xk], v[yk]
                    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    t = v[-1]
                    return area - t
                
                constraints.append({'type': 'ineq', 'fun': area_constraint})

    x0 = np.zeros(2 * N + 1)
    x0[:2 * N] = best_points_de.flatten()
    x0[-1] = best_val_de
    
    slsqp_bounds = []
    for i in range(N):
        slsqp_bounds.append((0, 1))
        slsqp_bounds.append((0, HALF_SQRT3))
    slsqp_bounds.append((0, 0.5))

    try:
        res_slsqp = minimize(
            slsqp_objective, 
            x0, 
            method='SLSQP', 
            bounds=slsqp_bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if res_slsqp.success:
            best_points_slsqp = res_slsqp.x[:2 * N].reshape(-1, 2)
            if is_inside_triangle(best_points_slsqp):
                return best_points_slsqp
    except Exception as e:
        pass
        
    return best_points_de
# EVOLVE-BLOCK-END
