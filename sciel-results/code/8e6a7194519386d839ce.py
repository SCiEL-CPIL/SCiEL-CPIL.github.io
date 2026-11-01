import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    # Precompute all triangle index triples as arrays for vectorization
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    tri_i = np.array([t[0] for t in triples], dtype=np.int32)
    tri_j = np.array([t[1] for t in triples], dtype=np.int32)
    tri_k = np.array([t[2] for t in triples], dtype=np.int32)
    
    def get_points(x):
        """
        x has 2*(N-2) elements: coordinates of points 2..N-1
        Points 0 and 1 are fixed at (0,0) and (1,0)
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        if N > 2:
            pts[2:, 0] = x[0::2]
            pts[2:, 1] = x[1::2]
        return pts
    
    def compute_areas_vectorized(pts):
        """Compute all triangle areas using full vectorization"""
        # pts[tri_i], pts[tri_j], pts[tri_k] gives arrays of shape (n_triples, 2)
        pi = pts[tri_i]
        pj = pts[tri_j]
        pk = pts[tri_k]
        # Cross product in 2D: (pj - pi) x (pk - pi)
        v1x = pj[:, 0] - pi[:, 0]
        v1y = pj[:, 1] - pi[:, 1]
        v2x = pk[:, 0] - pi[:, 0]
        v2y = pk[:, 1] - pi[:, 1]
        det = v1x * v2y - v1y * v2x
        areas = 0.5 * np.abs(det)
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
        shifted = areas - a_max
        exp_terms = np.exp(-beta * shifted)
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
    
    def initial_known_config():
        """
        Use a known good configuration for n=14 based on literature.
        Points roughly evenly distributed on a circle with slight perturbations
        to break symmetry and allow optimization to find better local minima.
        """
        x = np.zeros(2 * (N - 2))
        # Start with regular polygon but with small random perturbations
        rng = np.random.RandomState(12345)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Add small angular perturbations
        angle_perturb = rng.normal(0, 0.02, N)
        angles = angles + angle_perturb
        # Add small radial perturbations
        radii = 1.0 + rng.normal(0, 0.01, N)
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = radii[idx] * np.cos(angles[idx])
            x[2 * i + 1] = radii[idx] * np.sin(angles[idx])
        return x

    # Time management
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    # Correct bounds: 2*(N-2) dimensions
    dim = 2 * (N - 2)
    bounds = [(-3.0, 3.0)] * dim
    
    # Phase 1: Quick global search with differential evolution
    try:
        result_de = differential_evolution(
            raw_objective,
            bounds,
            seed=42,
            maxiter=20,
            popsize=12,
            tol=1e-3,
            init='sobol',
            polish=False,
            mutation=(0.5, 1.5),
            recombination=0.7
        )
        if result_de.fun < -best_score:
            best_x = result_de.x.copy()
            best_score = -result_de.fun
    except Exception:
        pass
    
    # Phase 2: Multi-start local optimization with adaptive annealing
    # Use a more aggressive and finer-grained beta schedule
    betas_schedule = [20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0]
    
    # Multiple diverse starting points
    start_points = []
    # Known good configuration
    start_points.append(initial_known_config())
    # Regular polygon based with various offsets
    for angle_offset in [0.0, 0.1, 0.3, 0.7, 1.0, 1.5, 2.0]:
        start_points.append(initial_circle(angle_offset, radius=1.0))
    # Elliptic configurations
    for ratio in [0.6, 0.75, 0.9, 0.95]:
        for angle_offset in [0.0, 0.5, 1.0]:
            start_points.append(initial_elliptic(ratio, angle_offset, radius=1.0))
    # Spiral configurations
    for tightness in [0.15, 0.3, 0.5]:
        start_points.append(initial_spiral(tightness, radius=1.0))
    
    # Add the DE result if available
    if best_x is not None:
        start_points.append(best_x.copy())
        for _ in range(3):
            jitter = np.random.normal(0, 0.1, size=dim)
            start_points.append(best_x + jitter)

    # Iterate through beta values (annealing) with restarts
    for beta in betas_schedule:
        if time.time() - start_time > time_limit - 15.0:
            break
        
        # For each starting point, run optimization
        for x0 in start_points:
            if time.time() - start_time > time_limit - 15.0:
                break
            
            # L-BFGS-B with correct bounds - primary method
            try:
                res = minimize(
                    logsumexp_min_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
            
            # BFGS without bounds - secondary method
            try:
                res2 = minimize(
                    logsumexp_min_objective,
                    x0,
                    args=(beta,),
                    method='BFGS',
                    options={'maxiter': 500, 'gtol': 1e-12}
                )
                
                if res2.fun < -best_score:
                    best_x = res2.x.copy()
                    best_score = -res2.fun
            except Exception:
                pass
        
        # Update start points with best found so far and add perturbations
        if best_x is not None:
            new_starts = [best_x.copy()]
            for _ in range(4):
                jitter = np.random.normal(0, 0.05, size=dim)
                new_starts.append(best_x + jitter)
            # Keep some diversity
            start_points = new_starts + start_points[:6]
    
    # Phase 3: Final high-precision refinement with very high beta
    if time.time() - start_time < time_limit - 10.0 and best_x is not None:
        for beta_final in [4000.0, 8000.0, 16000.0, 32000.0]:
            if time.time() - start_time > time_limit - 6.0:
                break
            
            try:
                res_final = minimize(
                    logsumexp_min_objective,
                    best_x,
                    args=(beta_final,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13}
                )
                
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead on raw objective for final polish
    if time.time() - start_time < time_limit - 4.0 and best_x is not None:
        try:
            res_nm = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_x = res_nm.x.copy()
                best_score = -res_nm.fun
        except Exception:
            pass
    
    # Phase 5: Final ultra-high precision polish
    if time.time() - start_time < time_limit - 2.0 and best_x is not None:
        try:
            res_polish = minimize(
                logsumexp_min_objective,
                best_x,
                args=(50000.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14}
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
