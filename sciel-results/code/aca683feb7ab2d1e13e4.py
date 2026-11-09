import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute triple indices for vectorized area calculation
    triples = list(itertools.combinations(range(N), 3))
    triples_arr = np.array(triples)
    p1_idx = triples_arr[:, 0]
    p2_idx = triples_arr[:, 1]
    p3_idx = triples_arr[:, 2]

    def get_score(x):
        """Compute the score: min triangle area / convex hull area"""
        points = x.reshape(N, 2)
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0  # Return 0 for degenerate cases
            
        if hull_area < 1e-9:
            return 0.0
            
        # Calculate Triangle Areas
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        p3 = points[p3_idx]
        
        # Area = 0.5 * | (p2-p1) x (p3-p1) |
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area

    def negative_score_flat(x):
        """Negative score for minimization (hard min)"""
        return -get_score(x)

    def soft_min(values, alpha):
        """
        Soft minimum approximation.
        values: array of areas
        alpha: temperature parameter (higher is sharper)
        """
        v = np.asarray(values)
        min_v = np.min(v)
        # Shift to prevent underflow/overflow in exp
        shifted = v - min_v
        exp_sum = np.sum(np.exp(-alpha * shifted))
        if exp_sum == 0 or not np.isfinite(exp_sum):
            return min_v
        return min_v - np.log(exp_sum) / alpha

    def objective_soft(x, alpha):
        """
        Objective function using soft-min for the minimum area.
        This makes the function smoother for gradient-based methods like L-BFGS-B.
        """
        points = x.reshape(N, 2)
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e10  # Large penalty for degenerate hulls
            
        if hull_area < 1e-9:
            return 1e10
            
        # Calculate Triangle Areas
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        p3 = points[p3_idx]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        sm = soft_min(areas, alpha)
        
        # We want to maximize sm / hull_area, so minimize -(sm / hull_area)
        return -(sm / hull_area)

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_x = x0.copy()
    best_score = get_score(x0)

    # Phase 1: Soft-Min Annealing with L-BFGS-B
    # Start with low alpha (smoother, finds basin) and increase to high alpha (sharp, refines min)
    alphas = [10.0, 50.0, 200.0, 1000.0]
    current_x = x0.copy()
    
    for alpha in alphas:
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(
                objective_soft, 
                current_x.flatten(), 
                args=(alpha,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
            )
            current_x = res.x.reshape(N, 2)
            
            current_score = get_score(current_x)
            if current_score > best_score:
                best_score = current_score
                best_x = current_x.copy()
                
        except Exception:
            pass

    # Phase 2: Hard-Min Polish with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            # Use the best x found so far as start for NM
            start_nm = best_x.flatten()
            res_nm = minimize(
                negative_score_flat, 
                start_nm, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            current_score = -res_nm.fun
            if current_score > best_score:
                best_score = current_score
                best_x = res_nm.x.reshape(N, 2)
        except Exception:
            pass

    # Phase 3: Random Restarts with Perturbations if time remains
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(123)
        # Try a few perturbations of the best solution
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            noise = rng.normal(0, 0.05, size=(N, 2))
            x_start = (best_x + noise).flatten()
            
            try:
                # Quick L-BFGS-B with high alpha (sharp)
                res = minimize(
                    objective_soft, 
                    x_start, 
                    args=(500.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                current_score = get_score(res.x)
                if current_score > best_score:
                    best_score = current_score
                    best_x = res.x.reshape(N, 2)
                    
                    # Polish this new best
                    if time.time() < deadline - 2.0:
                        res_nm = minimize(
                            negative_score_flat, 
                            best_x.flatten(), 
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                        )
                        current_score = -res_nm.fun
                        if current_score > best_score:
                            best_score = current_score
                            best_x = res_nm.x.reshape(N, 2)
            except Exception:
                pass

    return best_x
