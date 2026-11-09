import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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
        p2 = points[p1_idx]  # placeholder to avoid confusion
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
        """Negative score for minimization"""
        return -get_score(x)

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_x = x0.copy()
    best_score = get_score(x0)

    # Try optimization from regular polygon
    try:
        res = minimize(negative_score_flat, x0, method='L-BFGS-B', 
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14})
        
        if time.time() < deadline:
            current_score = -res.fun
            if current_score > best_score:
                best_x = res.x.copy()
                best_score = current_score
    except Exception:
        pass

    # Try Nelder-Mead from the best solution found
    if time.time() < deadline - 15.0:
        try:
            res_nm = minimize(negative_score_flat, best_x, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                     'adaptive': True})
            if time.time() < deadline:
                current_score = -res_nm.fun
                if current_score > best_score:
                    best_x = res_nm.x.copy()
                    best_score = current_score
        except Exception:
            pass

    # Try differential evolution for global search
    if time.time() < deadline - 30.0:
        try:
            # Define bounds for each coordinate
            bounds = [(-1, 1)] * (2 * N)
            
            # Use differential evolution with limited iterations
            de_res = differential_evolution(
                negative_score_flat, bounds,
                maxiter=100, tol=1e-10, seed=42,
                polish=True, init='sobol'
            )
            
            if time.time() < deadline:
                current_score = -de_res.fun
                if current_score > best_score:
                    best_x = de_res.x.copy()
                    best_score = current_score
        except Exception:
            pass

    # If time remains, try random restarts with perturbations
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        for restart in range(8):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the best solution found so far
            noise = rng.normal(0, 0.05, size=(N, 2))
            x_start = best_x + noise
            
            try:
                res = minimize(negative_score_flat, x_start, method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-14})
                
                if time.time() < deadline:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_x = res.x.copy()
                        best_score = current_score
            except Exception:
                pass

    return best_x
