import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11

def bary_to_cart(u, v):
    # Barycentric coordinates (u, v, 1-u-v) for triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    # x = u*0 + v*1 + (1-u-v)*0.5
    # y = u*0 + v*0 + (1-u-v)*(sqrt(3)/2)
    # x = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
    # y = (1-u-v)*sqrt(3)/2
    x = 0.5 * (1 + v - u)
    y = (1 - u - v) * (3 ** 0.5) / 2
    return x, y

def get_points_from_params(params):
    # params shape (N, 2)
    pts = np.zeros((N, 2))
    for i in range(N):
        u = params[i, 0]
        v = params[i, 1]
        x, y = bary_to_cart(u, v)
        pts[i] = [x, y]
    return pts

def min_triangle_area(pts):
    # Calculate minimum area of any triangle formed by 3 points
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    # Check if points are inside the triangle
    # u >= 0, v >= 0, u+v <= 1
    # params are u, v for each point
    for i in range(N):
        u = params[i * 2]
        v = params[i * 2 + 1]
        if u < 0 or v < 0 or u + v > 1:
            return -1.0  # Penalize invalid configurations
    
    pts = get_points_from_params(params.reshape(N, 2))
    return min_triangle_area(pts)

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Initial bounds for u, v
    bounds = [(0, 1)] * (2 * N)
    
    # Use Differential Evolution
    # popsize and maxiter tuned for speed within 120s
    # 22 dimensions, DE can be expensive.
    # Let's try popsize=15, maxiter=200. 
    # We can also use a vectorized objective if needed, but 165 triangles * 11 points is small enough.
    
    # To speed up, we can use a wrapper that checks time
    best_result = None
    best_value = -1.0
    
    def time_check_wrapper(params):
        nonlocal best_result, best_value
        if time.time() - start_time > timeout:
            return -1.0 # Stop search effectively by returning bad value, but DE might not stop immediately.
                        # Better to rely on maxiter or manual interruption if possible.
                        # Since we can't easily interrupt DE mid-run, we must set maxiter conservatively.
        
        val = objective(params)
        if val > best_value:
            best_value = val
            best_result = params.copy()
        return val

    # Estimate time per iteration.
    # One objective call takes ~1ms? 165 triangles is fast.
    # DE with popsize=15, maxiter=200 -> 3000 evaluations. 3s.
    # We can go higher. popsize=20, maxiter=500 -> 10000 evals. ~10s.
    # Let's aim for ~60s of compute time to be safe.
    
    try:
        res = differential_evolution(
            time_check_wrapper, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=20, 
            tol=1e-8, 
            init='sobol' # Sobol can be better for high dim? Or just 'latinhypercube'
        )
        if res.fun > best_value:
            best_value = res.fun
            best_result = res.x
    except Exception as e:
        pass

    # Local refinement with Nelder-Mead
    if best_result is not None and time.time() - start_time < timeout - 10:
        # Nelder-Mead on the 22D space
        # We need to handle bounds. Nelder-Mead doesn't handle bounds well.
        # We can use a penalty or just try it and see.
        # Better: use a constrained optimizer or just accept DE result if it's good.
        # Or use a simple coordinate ascent / local search.
        
        # Let's try a simple local search: random perturbations
        current_params = best_result.copy()
        current_val = best_value
        step_size = 0.01
        for _ in range(5000):
            if time.time() - start_time > timeout - 5:
                break
            # Random perturbation
            noise = np.random.normal(0, step_size, 2 * N)
            new_params = current_params + noise
            
            # Simple projection to bounds? No, objective returns -1 if invalid.
            val = objective(new_params)
            if val > current_val:
                current_params = new_params
                current_val = val
            else:
                # Adaptive step size
                if np.random.random() < 0.1:
                    step_size *= 0.9
                    if step_size < 1e-6:
                        step_size = 1e-6

        if current_val > best_value:
            best_value = current_val
            best_result = current_params

    # Convert best_result to points
    if best_result is None:
        # Fallback: random points
        pts = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random() * (1 - u)
            x, y = bary_to_cart(u, v)
            pts.append([x, y])
        return np.array(pts)
    
    pts = get_points_from_params(best_result.reshape(N, 2))
    return pts
