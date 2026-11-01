import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    """
    Maximize minimum triangle area / convex hull area for 14 points.
    Uses differential evolution for global search followed by
    multi-scale annealing with L-BFGS-B for refinement.
    """
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    # Precompute triangle indices for vectorized area computation
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    
    def get_points(x):
        """
        x has 2*(N-2) elements: coordinates of points 2..N-1
        Points 0 and 1 are fixed at (0,0) and (1,0)
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas_vectorized(pts):
        """Compute all triangle areas using vectorized operations"""
        pi = pts[tri_i]
        pj = pts[tri_j]
        pk = pts[tri_k]
        
        v1 = pj - pi
        v2 = pk - pi
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(dets)
        return areas
    
    def compute_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1e-10
    
    def objective_smooth(x, beta):
        """
        Smoothed objective: -min_area / hull_area using log-sum-exp.
        """
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        
        a_ref = np.min(areas)
        exp_terms = np.exp(-beta * (areas - a_ref))
        log_sum = -beta * a_ref + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)
    
    def objective_raw(x):
        """Raw objective: -min_area / hull_area"""
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def pts_to_x(pts):
        """Convert point array (with fixed pts[0], pts[1]) to parameter vector x"""
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x[2 * i] = pts[i + 2, 0]
            x[2 * i + 1] = pts[i + 2, 1]
        return x

    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    dim = 2 * (N - 2)
    bounds = [(-2.0, 2.0)] * dim
    
    # Phase 1: Global Search with Differential Evolution
    # Use a moderate beta for the smoothed objective to guide the search
    beta_de = 50.0
    
    if time.time() - start_time < time_limit - 30.0:
        try:
            # differential_evolution is a global optimizer
            # maxiter is the number of generations
            # tol is the convergence tolerance
            res_de = differential_evolution(
                objective_smooth,
                bounds,
                args=(beta_de,),
                seed=42,
                maxiter=200,
                tol=1e-6,
                popsize=15,
                init='sobol'
            )
            
            if res_de.fun < -best_score:
                best_x = res_de.x.copy()
                best_score = -res_de.fun
        except Exception as e:
            pass
    
    # If DE didn't find a good solution, fall back to a regular polygon
    if best_x is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i, 0] = 0.5 + 0.5 * np.cos(angles[i])
            pts[i, 1] = 0.5 + 0.5 * np.sin(angles[i])
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        best_x = pts_to_x(pts)
        best_score = -objective_raw(best_x)
        
    # Phase 2: Multi-scale Annealing Refinement with L-BFGS-B
    # Start from the DE solution and refine with increasing beta
    betas_refine = [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for beta in betas_refine:
        if time.time() - start_time > time_limit - 10.0:
            break
        
        try:
            res_refine = minimize(
                objective_smooth,
                best_x,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_refine.fun < -best_score:
                best_x = res_refine.x.copy()
                best_score = -res_refine.fun
        except Exception:
            pass
    
    # Phase 3: Final Polish with Raw Objective using Nelder-Mead
    # Nelder-Mead is gradient-free and good for final non-smooth optimization
    if time.time() - start_time < time_limit - 2.0:
        try:
            res_final = minimize(
                objective_raw,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_final.fun < -best_score:
                best_x = res_final.x.copy()
                best_score = -res_final.fun
        except Exception:
            pass

    return get_points(best_x)

# EVOLVE-BLOCK-END
