import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)

def triangle_areas(points):
    """
    Calculate the areas of all triangles formed by the points.
    """
    n = points.shape[0]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(
                    p1[0] * (p2[1] - p3[1]) + 
                    p2[0] * (p3[1] - p1[1]) + 
                    p3[0] * (p1[1] - p2[1])
                )
                areas.append(area)
    return np.array(areas)

def penalty_and_grad(points):
    """
    Calculate penalty and its gradient.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    
    pen = 0.0
    grad = np.zeros_like(points)
    
    # d1 = y < 0
    mask1 = d1 < 0
    if np.any(mask1):
        pen += np.sum(d1[mask1]**2)
        grad[mask1, 0] += 0.0
        grad[mask1, 1] += 2.0 * d1[mask1]
        
    # d2 = sqrt(3)*x - y < 0
    mask2 = d2 < 0
    if np.any(mask2):
        pen += np.sum(d2[mask2]**2)
        grad[mask2, 0] += 2.0 * d2[mask2] * SQRT3
        grad[mask2, 1] += 2.0 * d2[mask2] * (-1.0)
        
    # d3 = sqrt(3) - sqrt(3)*x - y < 0
    mask3 = d3 < 0
    if np.any(mask3):
        pen += np.sum(d3[mask3]**2)
        grad[mask3, 0] += 2.0 * d3[mask3] * (-SQRT3)
        grad[mask3, 1] += 2.0 * d3[mask3] * (-1.0)
        
    return pen, grad

def objective_and_grad(flat_params):
    points = flat_params.reshape(N, 2)
    
    # Check bounds for safety, though penalty handles it
    if np.any(points[:, 0] < -1) or np.any(points[:, 0] > 2) or np.any(points[:, 1] < -1) or np.any(points[:, 1] > 2):
        return 1e6, np.zeros_like(flat_params)

    pen, pen_grad = penalty_and_grad(points)
    
    # Calculate areas and find min
    areas = []
    min_area = np.inf
    min_idx = None
    n = N
    # We need to track indices to compute gradient
    # It's faster to compute in a loop and track min
    
    min_area_val = np.inf
    min_i, min_j, min_k = 0, 0, 0
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Cross product z
                z = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(z)
                if area < min_area_val:
                    min_area_val = area
                    min_i, min_j, min_k = i, j, k
                    min_z = z
                    
    # Compute gradient of min area
    # Objective is -min_area + 1000 * pen
    # Grad = -grad(min_area) + 1000 * grad(pen)
    
    grad_obj = 1000.0 * pen_grad.flatten()
    
    if min_area_val < np.inf:
        # Gradient of area for the min triangle
        # If z > 0, Area = 0.5 z. If z < 0, Area = -0.5 z.
        sign = 1.0 if min_z >= 0 else -1.0
        
        p1 = points[min_i]
        p2 = points[min_j]
        p3 = points[min_k]
        
        # Derivatives of z = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
        # d/dx1: -(y3-y1) + (y2-y1) = y2 - y3
        # d/dy1: -(x2-x1) + (x3-x1) = x3 - x2
        # d/dx2: (y3-y1)
        # d/dy2: -(x3-x1) = x1 - x3
        # d/dx3: -(y2-y1) = y1 - y2
        # d/dy3: (x2-x1)
        
        dz_dx1 = p2[1] - p3[1]
        dz_dy1 = p3[0] - p2[0]
        dz_dx2 = p3[1] - p1[1]
        dz_dy2 = p1[0] - p3[0]
        dz_dx3 = p1[1] - p2[1]
        dz_dy3 = p2[0] - p1[0]
        
        # d(Area)/d(point) = 0.5 * sign * d(z)/d(point)
        # Since Obj = -Area, Grad = -0.5 * sign * d(z)
        
        factor = -0.5 * sign
        
        grad_obj[min_i * 2 + 0] += factor * dz_dx1
        grad_obj[min_i * 2 + 1] += factor * dz_dy1
        grad_obj[min_j * 2 + 0] += factor * dz_dx2
        grad_obj[min_j * 2 + 1] += factor * dz_dy2
        grad_obj[min_k * 2 + 0] += factor * dz_dx3
        grad_obj[min_k * 2 + 1] += factor * dz_dy3
        
    val = -min_area_val + 1000.0 * pen
    return val, grad_obj

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple random restarts with L-BFGS-B
    for seed in range(50):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed)
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * SQRT3 / 2.0
        
        flat_init = init_points.flatten()
        
        try:
            result = minimize(
                objective_and_grad, 
                flat_init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'maxfev': 10000, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            min_area = np.min(triangle_areas(opt_points))
            pen, _ = penalty_and_grad(opt_points)
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 1000 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass

    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            bh_result = basinhopping(
                objective_and_grad,
                best_points.flatten(),
                minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 'maxiter': 50, 'maxfev': 5000},
                niter=10,
                seed=123
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_min_area = np.min(triangle_areas(bh_points))
            bh_pen, _ = penalty_and_grad(bh_points)
            
            if bh_pen < 1e-4:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 1000 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
