import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Precompute indices for combinations
    indices = list(itertools.combinations(range(N), 3))
    
    def check_validity(points):
        x = points[:, 0]
        y = points[:, 1]
        # Triangle vertices: (0,0), (1,0), (0.5, H)
        # Conditions:
        # 1. y >= 0
        # 2. y <= sqrt(3) * x  (line from (0,0) to (1,0)? No, line from (0,0) to (0.5, H) is y = sqrt(3)x)
        #    Wait, let's verify.
        #    V0=(0,0), V1=(1,0), V2=(0.5, H).
        #    Edge V0-V1: y=0.
        #    Edge V0-V2: slope = H/0.5 = 2H = sqrt(3). So y = sqrt(3)x.
        #    Edge V1-V2: slope = (H-0)/(0.5-1) = -sqrt(3). Line: y - 0 = -sqrt(3)(x-1) => y = -sqrt(3)x + sqrt(3).
        #    So:
        #    y >= 0
        #    y <= sqrt(3) x
        #    y <= -sqrt(3) x + sqrt(3)
        
        if np.any(y < -1e-9): return False
        if np.any(y > SQRT3 * x + 1e-9): return False
        if np.any(y > -SQRT3 * x + SQRT3 + 1e-9): return False
        return True

    def min_area(points):
        x = points[:, 0]
        y = points[:, 1]
        min_area = float('inf')
        for i, j, k in indices:
            area = 0.5 * abs(x[i]*(y[j]-y[k]) + x[j]*(y[k]-y[i]) + x[k]*(y[i]-y[j]))
            if area < min_area:
                min_area = area
        return min_area

    def objective_de(params):
        points = params.reshape(N, 2)
        if not check_validity(points):
            return 1e10
        ma = min_area(points)
        if ma < 1e-12:
            return 1e10
        return -ma

    # 1. Global Search with Differential Evolution
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, H))   # y
    
    # Use a moderate number of iterations to find a good basin
    # We have ~110s. DE is expensive.
    # Let's use popsize=15, maxiter=200.
    # If time is tight, we can reduce.
    
    # We will run DE, then take the best result and polish with L-BFGS-B using a smooth approximation.
    
    result_de = differential_evolution(
        objective_de, 
        bounds, 
        maxiter=300, 
        popsize=15, 
        tol=1e-7, 
        seed=42, 
        mutation=(0.5, 1.5), 
        recombination=0.9,
        polish=False # We will do our own polishing with a smooth objective
    )
    
    best_points = result_de.x.reshape(N, 2)
    best_score = -result_de.fun
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout - 20:
        # Not enough time for polishing, return best DE result
        # Ensure validity
        if not check_validity(best_points):
             # Fallback: generate random valid points
             return _random_valid_points(N)
        return best_points

    # 2. Local Refinement with Smooth Objective (Soft-Min)
    # The hard minimum is non-differentiable.
    # We can use a soft-min approximation: -log(sum(exp(-area_i / T))) * T
    # Or simply minimize the sum of negative powers?
    # Actually, L-BFGS-B on the hard min often works if initialized well, but can get stuck on kinks.
    # A better approach for "maximize min" is to introduce a variable t and maximize t
    # subject to area_i >= t for all i.
    # This is a convex optimization problem if we fix the points? No, area is bilinear in points.
    # So it's non-convex.
    
    # Let's try minimizing -min_area with L-BFGS-B first, but with a slight smoothing if it fails.
    # Or just use L-BFGS-B on the hard min. It often converges to a local optimum where the gradient is zero (or subgradient contains 0).
    
    def objective_lbfgs(params):
        points = params.reshape(N, 2)
        # We don't strictly need to check validity here if we stay close to valid points,
        # but L-BFGS-B might step out.
        # We can add a penalty if invalid.
        if not check_validity(points):
             # Calculate distance to boundary? Or just return large value.
             # Better: project back? L-BFGS-B doesn't support projection easily.
             # Let's use a penalty.
             x = points[:, 0]
             y = points[:, 1]
             pen = 0
             pen += np.sum(np.maximum(0, -y)**2)
             pen += np.sum(np.maximum(0, y - SQRT3*x)**2)
             pen += np.sum(np.maximum(0, y - (-SQRT3*x + SQRT3))**2)
             return 1e10 + pen
             
        ma = min_area(points)
        if ma < 1e-12:
            return 1e10
        return -ma

    # Try polishing from the best DE result
    try:
        res_lbfgs = minimize(
            objective_lbfgs, 
            best_points.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        
        if res_lbfgs.success or res_lbfgs.fun < result_de.fun:
            new_points = res_lbfgs.x.reshape(N, 2)
            if check_validity(new_points):
                best_points = new_points
                best_score = -res_lbfgs.fun
    except Exception:
        pass

    # Check time
    elapsed = time.time() - start_time
    if elapsed > timeout - 10:
        if not check_validity(best_points):
             return _random_valid_points(N)
        return best_points

    # 3. Multi-start Local Refinement
    # Generate a few random valid points and polish them, keep the best.
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
            
        # Generate random points in triangle
        pts = _random_valid_points(N)
        
        try:
            res_r = minimize(
                objective_lbfgs, 
                pts.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            if res_r.success or res_r.fun < result_de.fun: # Compare against original DE best
                 if check_validity(res_r.x.reshape(N, 2)):
                     if -res_r.fun > best_score:
                         best_points = res_r.x.reshape(N, 2)
                         best_score = -res_r.fun
        except Exception:
            pass

    if not check_validity(best_points):
        return _random_valid_points(N)
        
    return best_points

def _random_valid_points(n):
    points = []
    for _ in range(n):
        while True:
            u, v = np.random.rand(2)
            if u + v < 1:
                # Barycentric coordinates
                # V0=(0,0), V1=(1,0), V2=(0.5, H)
                # P = u*V0 + v*V1 + (1-u-v)*V2
                x = v * 1.0 + (1-u-v) * 0.5
                y = (1-u-v) * H
                points.append([x, y])
                break
    return np.array(points)
