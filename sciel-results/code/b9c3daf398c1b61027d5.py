import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    """
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 580  # Safety margin before 600s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if np.any(dists == 0):
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x, beta):
        """
        Minimize soft_max(log(d)) - soft_min(log(d))
        using logsumexp for numerical stability.
        """
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            return 1e10

        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10

        # Center the points to remove translational invariance
        p = p - np.mean(p, axis=0)
        
        # Normalize scale to remove scaling invariance
        # This helps with numerical stability and keeps the optimizer in a bounded region
        norm_sq = np.sum(p**2)
        if norm_sq < 1e-12:
            return 1e10 + 1.0 / norm_sq
        
        # We want to fix the scale. 
        # The metric is scale invariant. 
        # Fixing sum(p^2) = n means average norm is 1.
        p = p * np.sqrt(n / norm_sq)

        dists = pdist(p, metric='euclidean')
        
        min_dist = np.min(dists)
        if min_dist < 1e-12:
            return 1e10 + 1.0 / min_dist

        log_dists = np.log(dists)

        # Soft max/min of log_dists
        bm = beta * log_dists
        bmin = -beta * log_dists
        
        # logsumexp handles the stability
        soft_max_val = (1.0 / beta) * logsumexp(bm)
        soft_min_val = -(1.0 / beta) * logsumexp(bmin)
        
        return soft_max_val - soft_min_val

    # Initial configurations
    initializations = []
    
    # 1. Icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
    poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    initializations.append(np.vstack([ico_verts, poles]))
    
    # 2. Fibonacci sphere
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib_pts = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - (i / (n - 1)) * 2.0  # z goes from 1 to -1
        radius = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        fib_pts[i] = [x, y, z]
    initializations.append(fib_pts)

    # 3. Random points on sphere
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.randn(n, d)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        initializations.append(rand_pts)

    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    # Optimization loop
    # Use a schedule of increasing beta (sharpening)
    # Start with small beta (smooth), increase to large beta (sharp)
    beta_schedule = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > time_limit - 30:
            break
            
        try:
            current_pts = init_pts.copy()
            for beta_val in beta_schedule:
                if time.time() - start_time > time_limit - 10:
                    break
                    
                res = minimize(objective, current_pts.ravel(), method='L-BFGS-B', 
                               args=(beta_val,),
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        # Normalize to unit sphere or max dist 1? 
                        # The ratio is scale invariant, so we can just keep them.
                        # But scaling helps with numerical stability in next steps.
                        dists_opt = pdist(opt_points)
                        max_d = np.max(dists_opt)
                        if max_d > 1e-10:
                            opt_points = opt_points / max_d
                            
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                        # Update current_pts to continue from this solution
                        current_pts = opt_points.copy()
                        
        except Exception:
            pass

    # Final high-precision polish
    if time.time() - start_time < time_limit - 20:
        try:
            # Use very large beta for near-exact min/max
            for beta_val in [500.0, 1000.0, 2000.0]:
                if time.time() - start_time > time_limit - 10:
                    break
                res = minimize(objective, best_points.ravel(), method='L-BFGS-B', 
                               args=(beta_val,),
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        dists_opt = pdist(opt_points)
                        max_d = np.max(dists_opt)
                        if max_d > 1e-10:
                            opt_points = opt_points / max_d
                            
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
        except Exception:
            pass

    # Final safety check
    if not np.all(np.isfinite(best_points)):
        best_points = initializations[0].copy()
        
    return best_points
