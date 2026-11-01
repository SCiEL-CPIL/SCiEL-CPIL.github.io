import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_min_area(pts):
        """Compute minimum triangle area and hull area"""
        areas = []
        for i, j, k in itertools.combinations(range(N), 3):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas.append(0.5 * abs(det))
        min_area = min(areas)
        
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            hull_area = 1.0
        
        if hull_area < 1e-10:
            return 0.0, 1.0
        return min_area, hull_area
    
    def logsumexp_min(areas, beta):
        """Compute smooth approximation of minimum using log-sum-exp"""
        max_a = np.max(areas)
        shifted = areas - max_a
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        return -(max_a + log_sum) / beta
    
    def smoothed_objective(x, beta=200.0):
        """
        Smoothed objective using log-sum-exp approximation of min.
        Maximizes min_area / hull_area via smooth lower bound.
        """
        pts = get_points(x)
        
        # Vectorized triangle area computation
        idx = np.array(list(itertools.combinations(range(N), 3)))
        p_i = pts[idx[:, 0]]
        p_j = pts[idx[:, 1]]
        p_k = pts[idx[:, 2]]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(dets)
        
        smooth_min = logsumexp_min(areas, beta)
        
        # Hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 1.0  # High value = bad (we're minimizing)
        
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)
    
    def raw_objective(x):
        """Raw objective for final evaluation"""
        pts = get_points(x)
        min_area, hull_area = compute_min_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def initial_guess(seed_offset=0):
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = np.cos(angles[idx])
            x[2 * i + 1] = np.sin(angles[idx])
        return x
    
    best_score = -np.inf
    best_x = None
    
    start_time = time.time()
    time_limit = 110.0
    
    # Phase 1: Global search with differential evolution
    bounds = [(-3.0, 3.0)] * (2 * (N - 2))
    
    try:
        result_de = differential_evolution(
            raw_objective,
            bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-6,
            init='sobol'
        )
        best_x = result_de.x
        best_score = -result_de.fun
    except Exception:
        best_x = initial_guess(0)
        best_score = -raw_objective(best_x)
    
    if time.time() - start_time > time_limit:
        return get_points(best_x)
    
    # Phase 2: Multi-scale log-sum-exp refinement
    # Start with coarse smoothing, progressively refine
    betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_x = best_x.copy()
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        # Multiple restarts from perturbed initial points
        for restart in range(4):
            if time.time() - start_time > time_limit:
                break
            
            if restart == 0:
                x0 = current_x.copy()
            else:
                x0 = current_x + np.random.randn(len(current_x)) * 0.3 * (1.0 / (restart + 1))
            
            try:
                res = minimize(
                    smoothed_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.fun < -best_score:
                    best_x = res.x
                    best_score = -res.fun
                    current_x = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: High-precision local refinement with very high beta
    if time.time() - start_time < time_limit - 10:
        for beta in [2000.0, 5000.0]:
            if time.time() - start_time > time_limit:
                break
            
            try:
                res = minimize(
                    smoothed_objective,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if res.fun < -best_score:
                    best_x = res.x
                    best_score = -res.fun
            except Exception:
                pass
    
    # Phase 4: Final Nelder-Mead polish on raw objective
    if time.time() - start_time < time_limit - 5:
        try:
            res_local = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_local.fun < -best_score:
                best_x = res_local.x
                best_score = -res_local.fun
        except Exception:
            pass
    
    # Phase 5: One more round of smoothing with best found configuration
    if time.time() - start_time < time_limit - 2:
        try:
            res_final = minimize(
                smoothed_objective,
                best_x,
                args=(1000.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_final.fun < -best_score:
                best_x = res_final.x
                best_score = -res_final.fun
        except Exception:
            pass
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
