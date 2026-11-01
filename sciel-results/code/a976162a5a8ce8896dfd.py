import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial import ConvexHull
import itertools
import time

N = 14
DIM = 2 * N

def construct():
    # Precompute all triangle index triples
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    def get_points(x):
        """
        x has 2*N elements: coordinates of all N points
        """
        return x.reshape(N, 2)
    
    def compute_areas(pts):
        """Compute all triangle areas"""
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
            # Check if points are collinear or degenerate
            if np.allclose(pts, pts[0]):
                return 0.0
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0
    
    def objective(x, beta=100.0):
        """
        Objective function: -min_area / hull_area
        Uses log-sum-exp for smoothing.
        """
        pts = get_points(x)
        
        # Optional: Regularize scale to avoid drift
        # The problem is scale invariant. Adding a small penalty to norm helps
        # numerical stability and prevents the optimizer from wandering to infinity/zero.
        # We target a norm of roughly sqrt(N) * 1.0 (unit radius circle)
        norm_sq = np.sum(x**2)
        # Penalty is small so it doesn't distort the optimal shape significantly
        penalty = 1e-6 * (norm_sq - N)**2 
        
        areas = compute_areas(pts)
        hull_area = compute_hull_area(pts)
        
        if hull_area < 1e-10 or np.any(areas < 1e-15):
            return 1.0 + penalty
        
        # Log-sum-exp approximation of min
        # min(a_i) ~ -1/beta * log(sum(exp(-beta * a_i)))
        # To be numerically stable:
        a_min = np.min(areas)
        # If we just want the min, we can use it directly for high beta, 
        # but logsumexp is smoother.
        # For high beta, logsumexp converges to min.
        
        # Stable logsumexp:
        # L = log(sum(exp(-beta * a_i)))
        # = -beta * max(a_i) + log(sum(exp(-beta * (a_i - max(a_i)))))
        # Wait, we want min. 
        # min(a) = - max(-a)
        # -max(-a) = - ( -1/beta * log(sum(exp(-beta * (-(-a_i))))) ) ??
        # Let's use the standard formula for min:
        # min(a_i) ≈ -1/beta * log( sum( exp(-beta * a_i) ) )
        
        # Stable calculation:
        # Let m = min(a_i).
        # sum(exp(-beta * a_i)) = exp(-beta * m) * sum(exp(-beta * (a_i - m)))
        # log(sum(...)) = -beta * m + log(sum(exp(-beta * (a_i - m))))
        # -1/beta * log(sum(...)) = m - 1/beta * log(sum(exp(-beta * (a_i - m))))
        
        # This is very stable.
        m = np.min(areas)
        # If m is very small, the term 1/beta * log(...) might be significant if beta is small.
        # For large beta, it approaches m.
        
        # We can just use m if beta is large enough. 
        # But for smoothness during optimization, we use the approximation.
        
        shifted = areas - m
        exp_terms = np.exp(-beta * shifted)
        log_sum = np.log(np.sum(exp_terms))
        smooth_min = m - (1.0 / beta) * log_sum
        
        score = smooth_min / hull_area
        return -score + penalty

    def raw_objective(x):
        """
        Objective for final polish: exact min area / hull area
        """
        pts = get_points(x)
        areas = compute_areas(pts)
        hull_area = compute_hull_area(pts)
        
        if hull_area < 1e-10:
            return 1.0
        
        min_area = np.min(areas)
        return -min_area / hull_area

    def initial_regular_polygon():
        """
        Generate points on a unit circle
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = np.zeros(DIM)
        for i in range(N):
            x[2*i] = np.cos(angles[i])
            x[2*i+1] = np.sin(angles[i])
        return x

    start_time = time.time()
    time_limit = 115.0
    
    best_x = initial_regular_polygon()
    best_score = -raw_objective(best_x)
    
    # Phase 1: Basin Hopping for global optimization
    # Use a moderate beta for smoothness
    beta_bh = 50.0
    
    def bh_objective(x):
        return objective(x, beta=beta_bh)
    
    try:
        # basinhopping with L-BFGS-B
        # stepsize: size of random perturbations
        # niter: number of basin hopping iterations
        # We limit time by checking in the callback or just setting low niter
        
        # Estimate time per iteration. 
        # L-BFGS-B on 28 dims is fast.
        # Let's try 50 iterations of basin hopping.
        
        result_bh = basinhopping(
            bh_objective,
            x0=best_x,
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 500, 'ftol': 1e-10}},
            stepsize=0.5,
            niter=50,
            seed=42,
            disp=False
        )
        
        if result_bh.fun < best_score:
            best_score = result_bh.fun
            best_x = result_bh.x.copy()
            
    except Exception as e:
        pass

    # Phase 2: Refinement with increasing beta
    # Start from the best point found
    current_x = best_x.copy()
    
    betas = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        def ref_objective(x):
            return objective(x, beta=beta)
        
        try:
            res = minimize(
                ref_objective,
                current_x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                current_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final Polish with Raw Objective (Non-smooth)
    # Use Nelder-Mead or Powell which can handle non-smoothness better than gradient methods
    # Or just high beta L-BFGS-B.
    # Let's try Nelder-Mead on the raw objective for the last mile.
    
    if time.time() - start_time < time_limit - 2.0:
        try:
            res_nm = minimize(
                raw_objective,
                current_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                current_x = res_nm.x.copy()
        except Exception:
            pass

    # Phase 4: One more L-BFGS-B with very high beta to squeeze out precision
    if time.time() - start_time < time_limit - 1.0:
        try:
            res_final = minimize(
                lambda x: objective(x, beta=5000.0),
                current_x,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if res_final.fun < best_score:
                best_score = res_final.fun
                current_x = res_final.x.copy()
        except Exception:
            pass

    return get_points(current_x)
