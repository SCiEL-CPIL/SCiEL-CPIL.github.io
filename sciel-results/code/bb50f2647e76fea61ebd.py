import numpy as np
from scipy.optimize import minimize, differential_evolution
from math import log, pi
import time

def construct():
    n = 8
    start_time = time.time()
    max_time = 140.0
    
    # Regular octagon
    angles = np.linspace(0, 2 * pi, n, endpoint=False)
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    best_score = -np.inf
    best_points = x0.copy()
    
    def log_product(params):
        pts = params.reshape(n, 2)
        dist = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        mask = ~np.eye(n, dtype=bool)
        dists = dist[mask]
        if np.any(dists <= 0):
            return -np.inf
        return np.sum(np.log(dists))
    
    def score(params):
        lp = log_product(params)
        if lp == -np.inf:
            return -np.inf
        return np.exp(lp - 8 * log(8))
    
    def penalized_objective(params):
        lp = log_product(params)
        if lp == -np.inf:
            return 1e10
        pts = params.reshape(n, 2)
        dist = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        mask = ~np.eye(n, dtype=bool)
        dists = dist[mask]
        violations = np.maximum(0, dists - 2.0)
        penalty = 100.0 * np.sum(violations ** 2)
        return -lp + penalty
    
    # Phase 1: Global search with Differential Evolution
    # Bounds for x, y
    bounds = [(-1.1, 1.1)] * (2 * n)
    
    # Run DE for a fixed time
    de_time_limit = 100.0
    
    def time_check(iteration, xk, convergence):
        if time.time() - start_time > de_time_limit:
            return True
        return False

    try:
        # maxiter is high, but we stop based on time
        res_de = differential_evolution(
            penalized_objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            tol=1e-8,
            callback=time_check,
            polish=False # We will polish manually
        )
        if res_de.success or res_de.fun < 1e9:
            best_points_de = res_de.x
            s = score(best_points_de)
            if s > best_score:
                best_score = s
                best_points = best_points_de
    except Exception as e:
        pass
    
    # Phase 2: Polish with SLSQP
    if time.time() - start_time < max_time - 10:
        # Define constraints for SLSQP
        def constraints(params):
            pts = params.reshape(n, 2)
            dist = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            mask = ~np.eye(n, dtype=bool)
            dists = dist[mask]
            return 2.0 - dists
        
        cons = [{'type': 'ineq', 'fun': constraints}]
        bounds_slsqp = [(-1.1, 1.1)] * (2 * n)
        
        # Polish the best point from DE
        try:
            res_slsqp = minimize(
                penalized_objective, # Using penalized to be safe, or objective?
                best_points, 
                method='SLSQP', 
                bounds=bounds_slsqp, 
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res_slsqp.success:
                s = score(res_slsqp.x)
                if s > best_score:
                    best_score = s
                    best_points = res_slsqp.x
        except Exception:
            pass
        
        # Also polish the regular octagon in case DE failed
        if time.time() - start_time < max_time - 5:
            try:
                res_slsqp_oct = minimize(
                    penalized_objective,
                    x0, 
                    method='SLSQP', 
                    bounds=bounds_slsqp, 
                    constraints=cons,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_slsqp_oct.success:
                    s = score(res_slsqp_oct.x)
                    if s > best_score:
                        best_score = s
                        best_points = res_slsqp_oct.x
            except Exception:
                pass

    pts = best_points.reshape(n, 2)
    return [[float(p[0]), float(p[1])] for p in pts]
