import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def bary_to_cart(params):
    # params shape (2N,)
    # u_i, v_i for i in 0..N-1
    # w_i = 1 - u_i - v_i
    u = params[0::2]
    v = params[1::2]
    w = 1.0 - u - v
    
    # x = u*Ax + v*Bx + w*Cx = 0 + v*1 + w*0.5 = v + 0.5w
    # y = u*Ay + v*By + w*Cy = 0 + 0 + w*sqrt(3)/2
    x = v + 0.5 * w
    y = (np.sqrt(3)/2) * w
    
    return np.column_stack((x, y))

def compute_min_area(points):
    # points shape (N, 2)
    # Compute areas of all triangles
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # 2*Area = |(x2-x1)(y3-y1) - (y2-y1)(x3-x1)|
                area2 = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                areas.append(area2 * 0.5)
    return np.min(areas)

def exact_obj(params):
    points = bary_to_cart(params)
    return -compute_min_area(points)

def smooth_obj(params, beta):
    points = bary_to_cart(params)
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area2 = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                areas.append(area2 * 0.5)
    areas = np.array(areas)
    # Log-sum-exp of -beta * area
    # min(areas) approx -1/beta * log(sum(exp(-beta*areas)))
    # We want to maximize min area, so minimize -min area.
    # Objective: -min(areas)
    # Smooth approx: -(-1/beta * log(sum(exp(-beta*areas)))) = 1/beta * log(sum(exp(-beta*areas)))
    # Wait, log-sum-exp is usually for max.
    # min(x) = -max(-x)
    # max(-x) approx 1/beta * log(sum(exp(beta*(-x))))
    # So min(x) approx -1/beta * log(sum(exp(-beta*x)))
    # We want to minimize -min(x).
    # So objective = min(x) approx -1/beta * log(sum(exp(-beta*x)))
    # Let's check signs.
    # If areas are large, exp(-beta*area) is small.
    # If one area is small (bad), exp(-beta*area) is large.
    # Sum is dominated by the smallest area.
    # log(sum) approx -beta * min(areas).
    # -1/beta * log(sum) approx min(areas).
    # So minimizing this function minimizes the min area? No.
    # We want to MAXIMIZE min area.
    # So we should MINIMIZE -min area.
    # Smooth objective for minimization:
    # f(params) = - min(areas)
    # Approximation:
    # min(areas) = - max(-areas)
    # max(-areas) approx 1/beta * log(sum(exp(beta*(-areas))))
    # So min(areas) approx -1/beta * log(sum(exp(-beta*areas)))
    # Then -min(areas) approx 1/beta * log(sum(exp(-beta*areas)))
    # This function is positive and small when areas are large.
    # So minimizing this function maximizes the min area.
    
    # To avoid overflow/underflow in exp, use log-sum-exp trick.
    # log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    x = -beta * areas
    max_x = np.max(x)
    if max_x > 0: # shift to avoid overflow
        log_sum = max_x + np.log(np.sum(np.exp(x - max_x)))
    else:
        log_sum = np.log(np.sum(np.exp(x)))
    
    return (1.0 / beta) * log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_params = None
    best_score = -np.inf
    
    # Initial guesses
    # 1. Grid-like
    # 2. Random
    
    # Define bounds for barycentric coords
    # u, v >= 0, u+v <= 1
    # Box bounds: u in [0,1], v in [0,1]
    # The constraint u+v<=1 is handled by the geometry?
    # No, bary_to_cart assumes w = 1-u-v.
    # If u+v > 1, w < 0, point is outside.
    # So we must ensure u+v <= 1.
    # Differential evolution works on boxes.
    # We can use a transformation or just penalize.
    # Better: Use variables u, v, and clip/penalize if u+v > 1.
    # Or, use a different parameterization.
    # Let's stick to box [0,1]x[0,1] and penalize violations heavily.
    
    bounds = [(0, 1)] * (2 * N)
    
    def penalized_smooth_obj(params, beta):
        u = params[0::2]
        v = params[1::2]
        w = 1 - u - v
        if np.any(w < -1e-6):
            return 1e6 # Large penalty
        # Clip for safety in calculation
        w = np.clip(w, 0, 1)
        # Recalculate points with clipped w? No, just use original if valid.
        # If w < 0, it's invalid.
        return smooth_obj(params, beta)
        
    def penalized_exact_obj(params):
        u = params[0::2]
        v = params[1::2]
        w = 1 - u - v
        if np.any(w < -1e-6):
            return 1e6
        return exact_obj(params)

    # Run DE for a few seconds
    if time.time() < deadline - 30:
        res_de = differential_evolution(
            penalized_smooth_obj, 
            bounds, 
            args=(10.0,), # beta
            maxiter=100, 
            popsize=15,
            seed=42,
            tol=1e-4
        )
        if res_de.fun < 1e5:
            best_params = res_de.x
            best_score = -res_de.fun # Convert to max area
    
    # Polish with Nelder-Mead
    if best_params is not None and time.time() < deadline - 5:
        res_nm = minimize(
            penalized_exact_obj, 
            best_params, 
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res_nm.fun < 1e5:
            if -res_nm.fun > best_score:
                best_params = res_nm.x
                best_score = -res_nm.fun

    # If DE failed or didn't run, try a few random starts with NM
    if best_params is None or time.time() < deadline - 10:
        rng = np.random.RandomState(7)
        for _ in range(10):
            if time.time() > deadline - 10: break
            # Random valid barycentric coords
            u = rng.random(N)
            v = rng.random(N) * (1 - u) # Ensure u+v <= 1? No, v in [0, 1-u]
            # Correct random sampling in triangle:
            # u ~ U(0,1), v ~ U(0, 1-u)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            params = np.zeros(2*N)
            params[0::2] = u
            params[1::2] = v
            
            res_nm = minimize(
                penalized_exact_obj, 
                params, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res_nm.fun < 1e5:
                score = -res_nm.fun
                if score > best_score:
                    best_score = score
                    best_params = res_nm.x

    # Final conversion
    if best_params is None:
        # Fallback to seed-like random
        rng = np.random.RandomState(7)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_params = np.zeros(2*N)
        best_params[0::2] = u
        best_params[1::2] = v

    points = bary_to_cart(best_params)
    return points
