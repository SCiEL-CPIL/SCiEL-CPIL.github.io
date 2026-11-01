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
        """Compute all triangle areas efficiently using vectorization"""
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
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def compute_min_area(pts):
        """Compute minimum triangle area and hull area"""
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0, 1.0
        return min_area, hull_area
    
    def logsumexp_min_objective(x, beta=500.0):
        """
        Smoothed objective: -min_area / hull_area using log-sum-exp.
        min(a_i) ≈ -1/beta * log(sum(exp(-beta * a_i)))
        """
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        # Numerically stable log-sum-exp
        a_max = np.max(areas)
        if a_max < 1e-15:
            return 1.0
        
        # log(sum(exp(-beta * a))) = -beta * a_max + log(sum(exp(-beta * (a - a_max))))
        exp_terms = np.exp(-beta * (areas - a_max))
        log_sum = -beta * a_max + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)
    
    def raw_objective(x):
        """Raw objective: -min_area / hull_area"""
        pts = get_points(x)
        min_area, hull_area = compute_min_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def initial_circle(seed_offset=0.0, radius=1.0):
        """Generate initial points on a circle"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = radius * np.cos(angles[idx])
            x[2 * i + 1] = radius * np.sin(angles[idx])
        return x
    
    def initial_elliptic(ratio=0.7, seed_offset=0.0, radius=1.0):
        """Generate initial points on an ellipse"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = radius * np.cos(angles[idx])
            x[2 * i + 1] = radius * ratio * np.sin(angles[idx])
        return x

    def initial_spiral(tightness=0.3, radius=1.0):
        """Generate initial points on a spiral"""
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            idx = i + 2
            t = idx / (N - 2)
            angle = t * 2 * np.pi * 2  # Two full turns
            r = radius * (1 - tightness * t)
            x[2 * i] = r * np.cos(angle)
            x[2 * i + 1] = r * np.sin(angle)
        return x

    # Time management
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    # Correct bounds: 2*(N-2) dimensions
    dim = 2 * (N - 2)
    bounds = [(-3.0, 3.0)] * dim
    
    # Phase 1: Global search with differential evolution
    try:
        result_de = differential_evolution(
            raw_objective,
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
    
    # Phase 2: Multi-start local optimization with annealing
    betas_schedule = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]
    
    # Multiple diverse starting points
    start_points = []
    # Regular polygon based
    for angle_offset in [0.0, 0.2, 0.5, 0.8, 1.1, 1.5, 2.0]:
        start_points.append(initial_circle(angle_offset, radius=1.0))
    # Elliptic
    for ratio in [0.5, 0.65, 0.8, 0.9, 0.95]:
        for angle_offset in [0.0, 0.4, 0.8, 1.2]:
            start_points.append(initial_elliptic(ratio, angle_offset, radius=1.0))
    # Spiral
    for tightness in [0.2, 0.4, 0.6]:
        start_points.append(initial_spiral(tightness, radius=1.0))
    
    # Add the DE result if available
    if best_x is not None:
        start_points.append(best_x.copy())
        for _ in range(4):
            jitter = np.random.normal(0, 0.15, size=dim)
            start_points.append(best_x + jitter)

    # Iterate through beta values (annealing)
    for beta in betas_schedule:
        if time.time() - start_time > time_limit - 10.0:
            break
        
        for x0 in start_points:
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # L-BFGS-B with correct bounds
            try:
                res = minimize(
                    logsumexp_min_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 8000}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
            
            # BFGS without bounds
            try:
                res2 = minimize(
                    logsumexp_min_objective,
                    x0,
                    args=(beta,),
                    method='BFGS',
                    options={'maxiter': 400, 'gtol': 1e-11}
                )
                
                if res2.fun < -best_score:
                    best_x = res2.x.copy()
                    best_score = -res2.fun
            except Exception:
                pass
        
        # Update start points with best found so far
        if best_x is not None:
            new_starts = [best_x.copy()]
            for _ in range(3):
                jitter = np.random.normal(0, 0.08, size=dim)
                new_starts.append(best_x + jitter)
            start_points = new_starts + start_points[:4]
    
    # Phase 3: Final high-precision refinement
    if time.time() - start_time < time_limit - 8.0 and best_x is not None:
        for beta_final in [3000.0, 5000.0, 8000.0]:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            try:
                res_final = minimize(
                    logsumexp_min_objective,
                    best_x,
                    args=(beta_final,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead on raw objective
    if time.time() - start_time < time_limit - 3.0 and best_x is not None:
        try:
            res_nm = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_x = res_nm.x.copy()
                best_score = -res_nm.fun
        except Exception:
            pass
    
    # Phase 5: Final polish
    if time.time() - start_time < time_limit - 1.0 and best_x is not None:
        try:
            res_polish = minimize(
                logsumexp_min_objective,
                best_x,
                args=(15000.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            if res_polish.fun < -best_score:
                best_x = res_polish.x.copy()
                best_score = -res_polish.fun
        except Exception:
            pass
    
    if best_x is None:
        best_x = initial_circle(0.0)
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
