# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Returns 16 points in 2D that maximize the min/max distance ratio.
    Uses hexagonal lattice initialization with smooth surrogate optimization
    (log-sum-exp based soft-min/soft-max) and final polish pass.
    """
    start_time = time.time()
    deadline = start_time + 585  # Safety margin: 45s before 630s limit
    
    n = 16
    
    def get_score(pts):
        """Calculate R = (d_min / d_max)^2"""
        d = pdist(pts)
        if len(d) == 0:
            return 0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0
        return (d_min / d_max) ** 2
    
    def smooth_objective(params, beta_min=10.0, beta_max=10.0):
        """
        Smooth surrogate for minimizing -log(d_min/d_max) using log-sum-exp.
        We want to maximize d_min/d_max, so minimize -log(d_min) + log(d_max).
        
        soft_min(d) ≈ -log(sum(exp(-beta*d))) / beta
        soft_max(d) ≈ log(sum(exp(beta*d))) / beta
        
        We minimize: -soft_min + soft_max = log(sum(exp(-beta_min*d))) - log(sum(exp(beta_max*d))) + const
        """
        pts = params.reshape((n, 2))
        d = pdist(pts)
        
        # Scale to avoid overflow in exp
        d_min_ref = np.min(d)
        d_max_ref = np.max(d)
        
        # Soft min: -log(sum(exp(-beta*d))) / beta
        # For numerical stability, subtract max of -beta*d = -beta*d_min
        x_min = -beta_min * d
        x_min_shifted = x_min - np.max(x_min)
        soft_min = -np.log(np.sum(np.exp(x_min_shifted))) / beta_min
        
        # Soft max: log(sum(exp(beta*d))) / beta  
        # For numerical stability, subtract max of beta*d = beta*d_max
        x_max = beta_max * d
        x_max_shifted = x_max - np.max(x_max)
        soft_max = np.log(np.sum(np.exp(x_max_shifted))) / beta_max
        
        # We want to maximize d_min/d_max, so minimize (soft_max - soft_min)
        # This is equivalent to minimizing log(soft_max/soft_min) approximately
        return soft_max - soft_min
    
    def generate_hexagonal_initialization():
        """Generate points from hexagonal lattice subset"""
        u = np.array([1.0, 0.0])
        v = np.array([0.5, np.sqrt(3)/2])
        
        points_list = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * u + j * v
                dist_sq = np.dot(p, p)
                points_list.append((dist_sq, p))
        
        points_list.sort(key=lambda x: x[0])
        selected_points = np.array([p[1] for p in points_list[:16]])
        return selected_points
    
    def generate_random_initialization(seed):
        """Generate random points in unit square"""
        rng = np.random.RandomState(seed)
        return rng.rand(n, 2)
    
    # Generate multiple initializations
    initializations = [generate_hexagonal_initialization()]
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(generate_random_initialization(seed))
    
    best_pts = None
    best_score = -np.inf
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30:  # Need 30s for final polish
            break
        
        # Flatten for optimization
        x0 = init_pts.flatten()
        
        # Phase 1: Coarse optimization with moderate beta
        try:
            result = minimize(
                smooth_objective,
                x0,
                args=(5.0, 5.0),
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
            )
            if time.time() > deadline - 30:
                break
        except Exception:
            continue
        
        # Phase 2: Fine optimization with higher beta
        try:
            result2 = minimize(
                smooth_objective,
                result.x,
                args=(20.0, 20.0),
                method='Powell',
                options={'maxiter': 300, 'ftol': 1e-8, 'disp': False}
            )
            if time.time() > deadline - 30:
                break
        except Exception:
            result2 = result
        
        candidate_pts = result2.x.reshape((n, 2))
        score = get_score(candidate_pts)
        
        if score > best_score:
            best_score = score
            best_pts = candidate_pts.copy()
    
    if best_pts is None:
        # Fallback to hexagonal lattice
        best_pts = generate_hexagonal_initialization()
        best_score = get_score(best_pts)
    
    # Final high-precision polish: direct optimization on the actual objective
    # using very fine local search
    if time.time() < deadline - 10:
        # Use Nelder-Mead on the actual score function for final polish
        def neg_score(params):
            pts = params.reshape((n, 2))
            return -get_score(pts)
        
        try:
            final_result = minimize(
                neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False}
            )
            final_pts = final_result.x.reshape((n, 2))
            final_score = get_score(final_pts)
            
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
        except Exception:
            pass
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = generate_hexagonal_initialization()
    
    return best_pts

# EVOLVE-BLOCK-END
