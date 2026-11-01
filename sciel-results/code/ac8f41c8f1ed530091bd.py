import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    # Precompute all triangle index triples
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    def get_points_normalized(x):
        """
        x has 2*N elements.
        Returns points centered and scaled to unit norm (sum of squares = 1).
        """
        pts = x.reshape(N, 2)
        # Center
        pts = pts - np.mean(pts, axis=0, keepdims=True)
        # Scale
        norm = np.sqrt(np.sum(pts**2))
        if norm < 1e-12:
            # Fallback if all points are same
            return np.zeros((N, 2))
        pts = pts / norm
        return pts

    def compute_areas_vectorized(pts):
        """Compute all triangle areas efficiently"""
        areas = np.zeros(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas[idx] = 0.5 * abs(det)
        return areas

    def compute_hull_area(pts):
        """Compute convex hull area"""
        try:
            # ConvexHull requires at least 3 non-collinear points
            if len(pts) < 3:
                return 0.0
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0

    def smooth_objective(x, beta):
        """
        Objective: - (logsumexp(-beta * areas) / beta) / hull_area
        We minimize this, which is equivalent to maximizing min_area / hull_area.
        """
        pts = get_points_normalized(x)
        areas = compute_areas_vectorized(pts)
        
        # Handle degenerate cases
        if np.all(areas < 1e-12):
            return 1.0
        
        a_max = np.max(areas)
        if a_max < 1e-15:
            return 1.0
        
        # Stable log-sum-exp for min
        # min(a) approx -1/beta * log(sum(exp(-beta * a)))
        # log(sum(exp(-beta * a))) = -beta * a_max + log(sum(exp(-beta * (a - a_max))))
        exp_terms = np.exp(-beta * (areas - a_max))
        log_sum = -beta * a_max + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-12:
            return 1.0
        
        return -(smooth_min / hull_area)

    def raw_objective(x):
        """
        Objective for final evaluation: - min_area / hull_area
        """
        pts = get_points_normalized(x)
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-12:
            return 1.0
        return -(min_area / hull_area)

    # Time management
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    best_score = -np.inf
    best_x = None
    
    # Initial guess: Regular polygon
    # We need to initialize x such that get_points_normalized gives a polygon.
    # Since normalization happens inside, we can just use the polygon coords.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    init_x = init_pts.flatten()
    
    # Phase 1: Global Search with Differential Evolution
    # Bounds: Since we normalize internally, the magnitude of x doesn't matter much,
    # but DE needs bounds. We can use [-10, 10].
    bounds = [(-10.0, 10.0)] * (2 * N)
    
    try:
        # Use a smaller population and iterations to save time, relying on local opt later
        # DE is expensive.
        result_de = differential_evolution(
            lambda x: smooth_objective(x, beta=100.0), # Moderate beta for DE
            bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-4,
            init='sobol',
            polish=True,
            mutation=(0.5, 1.5),
            recombination=0.7
        )
        if result_de.fun < -best_score:
            best_x = result_de.x.copy()
            best_score = -result_de.fun
    except Exception:
        pass

    # If DE failed or didn't find anything, use init
    if best_x is None:
        best_x = init_x.copy()
        best_score = -smooth_objective(best_x, 100.0)

    # Phase 2: Local Optimization with Annealing Beta
    # Start with low beta (smooth) and increase.
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    current_x = best_x.copy()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Use L-BFGS-B
        try:
            res = minimize(
                smooth_objective,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                current_x = res.x.copy()
                best_x = res.x.copy()
        except Exception:
            pass
            
        # Try a few random restarts from best_x with jitter if time permits
        if time.time() - start_time < time_limit - 15.0:
             for _ in range(3):
                 if time.time() - start_time > time_limit - 10.0:
                     break
                 jitter = np.random.normal(0, 0.1, size=2*N)
                 x_try = best_x + jitter
                 try:
                     res_r = minimize(
                         smooth_objective,
                         x_try,
                         args=(beta,),
                         method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                     )
                     if res_r.fun < -best_score:
                         best_score = -res_r.fun
                         best_x = res_r.x.copy()
                         current_x = res_r.x.copy()
                 except Exception:
                     pass

    # Phase 3: Final Polish with Raw Objective (Nelder-Mead or L-BFGS on raw)
    # Raw objective is non-smooth, so Nelder-Mead or SLSQP might be better.
    # But we can use L-BFGS on a very high beta smoothed version.
    if time.time() - start_time < time_limit - 2.0 and best_x is not None:
        try:
            res_final = minimize(
                smooth_objective,
                best_x,
                args=(5000.0,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_final.fun < -best_score:
                best_score = -res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass

    # One last check with raw objective using Nelder-Mead (derivative-free, handles kinks)
    if time.time() - start_time < time_limit - 1.0 and best_x is not None:
        try:
            res_nm = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass

    # Return the points
    # The evaluator normalizes by convex hull area, so returning normalized points is fine.
    # But to be safe, return the points as they are in the best_x (which are normalized in get_points_normalized)
    # Wait, get_points_normalized returns the normalized points.
    # The evaluator will take these points, compute hull area, and divide min triangle area by it.
    # Since our points are already normalized (unit sum of squares), the hull area will be some value < 1.
    # This is fine.
    
    return get_points_normalized(best_x)

# EVOLVE-BLOCK-END
