import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def get_points_from_params(params):
        # params is array of shape (8,) -> 4 points (x, y)
        # Returns array of shape (8, 2)
        v = params.reshape(4, 2)
        # Centrally symmetric: v and -v
        p = np.vstack([v, -v])
        return p

    def neg_log_score(params):
        """
        Returns negative log of the scaled product of distances.
        We minimize this to maximize the score.
        Score ~ log(Prod(d_ij)) - 56 * log(D/2)
        """
        p = get_points_from_params(params)
        
        # Calculate pairwise distances
        # p is (8, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Check for duplicates (off-diagonal min distance)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist < 1e-6:
            return 1e10  # Large positive value (bad for minimization)
        
        # Diameter
        D = np.max(dists)
        if D < 1e-6:
            return 1e10
        
        # Log product of ordered pairs (i != j)
        # dists has 0 on diagonal. We want sum of log(dists[i,j]) for i != j.
        # Replace 0 with 1 so log is 0.
        dists_safe = np.where(dists > 0, dists, 1.0)
        log_prod = np.sum(np.log(dists_safe))
        
        # Subtract 56 * log(D/2)
        # 56 is number of ordered pairs (8 * 7)
        val = log_prod - 56.0 * np.log(D / 2.0)
        
        return -val

    # Differential Evolution
    bounds = [(-10.0, 10.0)] * 8
    
    best_x = None
    best_val = np.inf

    try:
        # Use a seed for reproducibility if needed, but random is fine
        res = differential_evolution(
            neg_log_score, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8, 
            polish=False
        )
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 20:
        if best_x is None:
            from math import cos, sin, pi
            return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        else:
            p = get_points_from_params(best_x)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            D = np.max(dists)
            if D > 0:
                p = p * (2.0 / D)
            return [[float(x), float(y)] for x, y in p]

    # Polish with Local Optimizer
    if best_x is not None:
        try:
            res_polish = minimize(
                neg_log_score, 
                best_x, 
                method='Powell', 
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res_polish.fun < best_val:
                best_x = res_polish.x
                best_val = res_polish.fun
        except Exception:
            pass

    # If time remains, try another DE run with different seed?
    # Or just return the best.
    # Given the speed, we could run DE again.
    if time.time() < deadline - 5:
        try:
            res2 = differential_evolution(
                neg_log_score, 
                bounds, 
                seed=123, 
                maxiter=50, 
                popsize=20, 
                tol=1e-8, 
                polish=False
            )
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
                # Polish this new best
                res_polish2 = minimize(
                    neg_log_score, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 100, 'ftol': 1e-12, 'xtol': 1e-12}
                )
                if res_polish2.fun < best_val:
                    best_x = res_polish2.x
                    best_val = res_polish2.fun
        except Exception:
            pass

    # Final check and scaling
    if best_x is None:
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    p = get_points_from_params(best_x)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    D = np.max(dists)
    
    if D < 1e-9:
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
    p = p * (2.0 / D)
    
    return [[float(x), float(y)] for x, y in p]
