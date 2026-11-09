import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n = 8
    # Number of pairs
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    num_pairs = len(pairs)
    
    def get_distances(points):
        # points is 2D array of shape (n, 2)
        # returns array of distances for all pairs
        dists = np.zeros(num_pairs)
        for k, (i, j) in enumerate(pairs):
            diff = points[i] - points[j]
            dists[k] = np.sqrt(diff[0]**2 + diff[1]**2)
        return dists

    def objective_and_constraints(points):
        # points is 1D array of shape (2n,)
        P = points.reshape(n, 2)
        dists = get_distances(P)
        
        # Avoid log(0)
        dists_safe = np.maximum(dists, 1e-10)
        
        # Objective: minimize negative log product
        obj = -np.sum(np.log(dists_safe))
        
        # Constraints: d_ij <= 2
        # scipy minimize expects constraints as dict or list of dicts
        # Inequality constraints: fun(x) >= 0
        # So we want 2 - d_ij >= 0
        cons = []
        for k in range(num_pairs):
            # We can't easily return a list of constraints from the objective function 
            # if we use the dict format in minimize. 
            # Better to define the constraints function separately or use the 'cons' list in minimize call.
            pass
            
        return obj

    # Redefine to fit scipy.optimize.minimize structure
    # Use penalty method: add large penalty if any distance > 2
    def obj_func(points):
        P = points.reshape(n, 2)
        dists = get_distances(P)
        dists_safe = np.maximum(dists, 1e-10)
        base_obj = -np.sum(np.log(dists_safe))
        # Penalty for violating diameter constraint
        violation = np.maximum(dists - 2.0, 0.0)
        penalty = 1000.0 * np.sum(violation**2)
        return base_obj + penalty

    def constraint_func(points):
        P = points.reshape(n, 2)
        dists = get_distances(P)
        # Return an array of values, each must be >= 0
        # We can return a single array if we use type='ineq' and the function returns an array?
        # No, scipy expects a scalar or we need to define each constraint separately.
        # Defining 28 separate constraint functions is inefficient but doable.
        # Alternatively, we can use a penalty method or just use 'SLSQP' with a list of dicts.
        # Let's create the list of constraint dicts outside the loop for efficiency.
        pass

    # No explicit constraints - using penalty method
    # Bounds for coordinates: keep within [-2, 2] to help with scaling
    bounds = [(-2.0, 2.0)] * (2 * n)

    # Initial guesses
    best_points = None
    best_score = -np.inf
    
    # Guess 1: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    p0_octagon = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Guess 2: Random points in unit disk
    np.random.seed(42)
    guesses = [p0_octagon]
    for _ in range(15):
        # Random points
        r = np.random.uniform(0, 1, n)
        theta = np.random.uniform(0, 2*np.pi, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # Scale to diameter approx 2
        # Just use them as is, optimizer will scale
        p0 = np.column_stack([x, y]).flatten()
        guesses.append(p0)
        
    # Guess 3: Two squares (rotated)
    angles_sq1 = 2 * np.pi * np.arange(4) / 4
    angles_sq2 = angles_sq1 + np.pi/4
    p_sq1 = np.column_stack([np.cos(angles_sq1), np.sin(angles_sq1)])
    p_sq2 = np.column_stack([np.cos(angles_sq2), np.sin(angles_sq2)])
    # Combine
    p0_squares = np.vstack([p_sq1, p_sq2]).flatten()
    guesses.append(p0_squares)
    
    # Guess 4: Perturbed octagon with random phase shifts
    for _ in range(5):
        angles_pert = angles + np.random.uniform(-0.3, 0.3, n)
        radii_pert = 1.0 + np.random.uniform(-0.2, 0.2, n)
        p0_pert = np.column_stack([radii_pert * np.cos(angles_pert), 
                                    radii_pert * np.sin(angles_pert)]).flatten()
        guesses.append(p0_pert)

    for p0 in guesses:
        if time.time() > deadline - 15.0:
            break
            
        try:
            # First pass: SLSQP with penalty
            res = minimize(
                obj_func, 
                p0, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < 1e6:
                P_opt = res.x.reshape(n, 2)
                # Check if all points are distinct
                if np.all(np.linalg.norm(np.diff(P_opt, axis=0), axis=1) > 1e-6):
                    # Calculate score
                    dists = get_distances(P_opt)
                    if np.all(dists > 0):
                        # Normalize to diameter 2
                        d_max = np.max(dists)
                        if d_max > 0:
                            scale = 2.0 / d_max
                            P_scaled = P_opt * scale
                            dists_scaled = dists * scale
                            log_score = np.sum(np.log(dists_scaled))
                            # The actual score is exp(log_score) / 8^8
                            # We just compare log_score
                            if log_score > best_score:
                                best_score = log_score
                                best_points = P_scaled
        except Exception as e:
            continue
    
    # Final high-precision polish pass if we have a good starting point
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            p0_polish = best_points.flatten()
            res_polish = minimize(
                obj_func, 
                p0_polish, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            if res_polish.success or res_polish.fun < 1e6:
                P_polish = res_polish.x.reshape(n, 2)
                if np.all(np.linalg.norm(np.diff(P_polish, axis=0), axis=1) > 1e-6):
                    dists_polish = get_distances(P_polish)
                    if np.all(dists_polish > 0):
                        d_max_polish = np.max(dists_polish)
                        if d_max_polish > 0:
                            scale_polish = 2.0 / d_max_polish
                            P_scaled_polish = P_polish * scale_polish
                            dists_scaled_polish = dists_polish * scale_polish
                            log_score_polish = np.sum(np.log(dists_scaled_polish))
                            if log_score_polish > best_score:
                                best_score = log_score_polish
                                best_points = P_scaled_polish
        except Exception as e:
            pass

    # If optimization failed or didn't improve, fall back to regular octagon
    if best_points is None:
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure return type is list of lists of floats
    return [[float(p[0]), float(p[1])] for p in best_points]
