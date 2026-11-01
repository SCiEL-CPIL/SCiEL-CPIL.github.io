import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    """
    Maximize minimum triangle area / convex hull area for 14 points.
    Uses Differential Evolution for global search and L-BFGS-B for local refinement.
    """
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    # Precompute triangle indices
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
            # Check for degenerate cases
            if np.allclose(pts, pts[0]):
                return 0.0
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0
    
    def objective_raw(x):
        """
        Raw objective: -min_area / hull_area
        Returns a positive value to minimize (negative of the score).
        """
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        
        if hull_area < 1e-10:
            return 1e10 # Large penalty for degenerate hull
            
        score = min_area / hull_area
        return -score
    
    def objective_smooth(x, beta):
        """
        Smoothed objective for gradient-based methods.
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
    
    def pts_to_x(pts):
        """Convert point array to parameter vector x"""
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x[2 * i] = pts[i + 2, 0]
            x[2 * i + 1] = pts[i + 2, 1]
        return x
    
    # Time management
    start_time = time.time()
    time_limit = 110.0 # Leave some buffer
    
    dim = 2 * (N - 2)
    
    # Define bounds for DE
    # The points should be within a reasonable range relative to the fixed points (0,0) and (1,0)
    # A square [-2, 2] x [-2, 2] should cover most interesting configurations
    bounds = [(-2.0, 2.0)] * dim
    
    best_score = -np.inf
    best_x = None
    
    # Phase 1: Global Search with Differential Evolution
    # DE is robust and doesn't require gradients.
    # We minimize -score.
    try:
        de_result = differential_evolution(
            objective_raw,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            atol=1e-10,
            polish=True, # Use L-BFGS-B to polish the result
            init='sobol'
        )
        
        if de_result.success or de_result.fun < -best_score:
            best_x = de_result.x.copy()
            best_score = -de_result.fun
    except Exception as e:
        pass
    
    # Check time
    if time.time() - start_time > time_limit:
        # Fallback if DE took too long (should not happen with limits)
        if best_x is None:
             # Fallback to regular polygon
             angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
             pts = np.zeros((N, 2))
             for i in range(N):
                 pts[i, 0] = 0.5 + 0.5 * np.cos(angles[i])
                 pts[i, 1] = 0.5 + 0.5 * np.sin(angles[i])
             pts[0] = [0.0, 0.0]
             pts[1] = [1.0, 0.0]
             best_x = pts_to_x(pts)
        return get_points(best_x)
    
    # Phase 2: Local Refinement with L-BFGS-B using Smoothed Objective
    # We try a few different beta values to refine the solution found by DE.
    # DE might have found a good local minimum, but L-BFGS-B can polish it to high precision.
    
    if best_x is not None:
        # Try multiple beta values for refinement
        betas = [100.0, 500.0, 1000.0, 5000.0]
        
        current_best_x = best_x.copy()
        current_best_score = best_score
        
        for beta in betas:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            try:
                # Start from the current best
                res = minimize(
                    objective_smooth,
                    current_best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                # Evaluate the raw score of the refined point
                raw_score = -objective_raw(res.x)
                if raw_score > current_best_score:
                    current_best_score = raw_score
                    current_best_x = res.x.copy()
            except Exception:
                pass
        
        best_x = current_best_x
        best_score = current_best_score
        
        # Phase 3: Final polish with raw objective and Nelder-Mead or L-BFGS-B
        # L-BFGS-B on raw objective might struggle with non-differentiability.
        # Let's try one more L-BFGS-B with a very high beta.
        if time.time() - start_time < time_limit - 2.0:
            try:
                res_final = minimize(
                    objective_smooth,
                    best_x,
                    args=(10000.0,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                raw_score = -objective_raw(res_final.x)
                if raw_score > best_score:
                    best_x = res_final.x.copy()
                    best_score = raw_score
            except Exception:
                pass
    
    # Fallback if something went wrong
    if best_x is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i, 0] = 0.5 + 0.5 * np.cos(angles[i])
            pts[i, 1] = 0.5 + 0.5 * np.sin(angles[i])
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        best_x = pts_to_x(pts)
        
    return get_points(best_x)
