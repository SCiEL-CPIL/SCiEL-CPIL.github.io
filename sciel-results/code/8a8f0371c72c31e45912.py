import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    triple_indices = np.array(triples)
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts

    def compute_areas(pts):
        # Vectorized area calculation
        # pts is (N, 2)
        # triple_indices is (M, 3)
        p_i = pts[triple_indices[:, 0]]
        p_j = pts[triple_indices[:, 1]]
        p_k = pts[triple_indices[:, 2]]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(dets)

    def compute_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0

    def objective_smooth(x, beta):
        pts = get_points(x)
        areas = compute_areas(pts)
        
        # Log-sum-exp approximation of min
        # min(a) ~ -1/beta * log(sum(exp(-beta * a)))
        # Stable version:
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        # Shift by min to avoid overflow/underflow in exp
        # Actually, we want to approximate MIN.
        # -logsumexp(-beta * areas) / beta
        # Let m = min(areas).
        # sum(exp(-beta * (a_i - m))) = exp(-beta*a_1+m) + ... 
        # log(sum(exp(-beta*(a_i - m)))) = log(exp(0) + exp(-beta(a_2-m)) + ...)
        # This is stable if m is the min.
        
        shifted = areas - a_min
        exp_terms = np.exp(-beta * shifted)
        log_sum = np.log(np.sum(exp_terms))
        smooth_min = a_min - log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)

    def objective_raw(x):
        pts = get_points(x)
        areas = compute_areas(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)

    def regular_polygon(offset=0.0, radius=1.0):
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = radius * np.cos(angles[idx])
            x[2 * i + 1] = radius * np.sin(angles[idx])
        return x

    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    dim = 2 * (N - 2)
    bounds = [(-3.0, 3.0)] * dim
    
    # Generate diverse starting points
    starts = []
    # Regular polygons with various rotations
    for i in range(10):
        starts.append(regular_polygon(offset=i * 0.2, radius=1.0))
        
    # Ellipses
    for ratio in [0.5, 0.7, 0.9]:
        for offset in [0.0, 0.5, 1.0]:
            x = np.zeros(dim)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + offset
            for i in range(N - 2):
                idx = i + 2
                x[2 * i] = np.cos(angles[idx])
                x[2 * i + 1] = ratio * np.sin(angles[idx])
            starts.append(x)
            
    # Random points in convex region (e.g. unit circle)
    rng = np.random.default_rng(42)
    for _ in range(10):
        x = rng.uniform(-1, 1, size=dim)
        starts.append(x)

    # Annealing schedule for beta
    # Start with low beta (smooth, easy to optimize) and increase (sharp, finds true min)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    # We will iterate through starts and betas.
    # To save time, we can use a greedy approach: 
    # 1. Optimize all starts with beta=10
    # 2. Take the best few, optimize with beta=50
    # 3. Take the best few, optimize with beta=100
    # etc.
    
    current_candidates = starts
    
    for beta in betas:
        if time.time() - start_time > time_limit - 10.0:
            break
            
        next_candidates = []
        
        for x0 in current_candidates:
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Try trust-constr
            try:
                res = minimize(
                    objective_smooth,
                    x0,
                    args=(beta,),
                    method='trust-constr',
                    bounds=bounds,
                    options={'maxiter': 500, 'gtol': 1e-8, 'verbose': 0}
                )
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
                next_candidates.append(res.x)
            except Exception:
                pass
                
            # Try L-BFGS-B as a backup/alternative
            try:
                res2 = minimize(
                    objective_smooth,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res2.fun < -best_score:
                    best_x = res2.x.copy()
                    best_score = -res2.fun
                next_candidates.append(res2.x)
            except Exception:
                pass
        
        # Keep top 5 candidates for next beta
        if not next_candidates:
            break
            
        # Evaluate raw score for candidates to rank them
        scored_candidates = []
        for xc in next_candidates:
            score = -objective_raw(xc)
            scored_candidates.append((score, xc))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        current_candidates = [xc for s, xc in scored_candidates[:5]]
        
        # If best_x is in candidates, ensure it's kept
        if best_x is not None:
            # Check if best_x is close to any candidate, if not, add it
            # Actually, since we updated best_x during optimization, it should be in next_candidates
            # But let's be safe.
            pass

    # Final refinement with very high beta on the best solution
    if best_x is not None and time.time() - start_time < time_limit - 5.0:
        for beta_final in [5000.0, 10000.0]:
            try:
                res_final = minimize(
                    objective_smooth,
                    best_x,
                    args=(beta_final,),
                    method='trust-constr',
                    bounds=bounds,
                    options={'maxiter': 1000, 'gtol': 1e-10}
                )
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
                
    # Final raw polish
    if best_x is not None and time.time() - start_time < time_limit - 2.0:
        try:
            res_raw = minimize(
                objective_raw,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_raw.fun < -best_score:
                best_x = res_raw.x.copy()
                best_score = -res_raw.fun
        except Exception:
            pass

    if best_x is None:
        best_x = regular_polygon(0.0)
        
    return get_points(best_x)
