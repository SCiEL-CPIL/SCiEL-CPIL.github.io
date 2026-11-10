import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 585  # Safety margin before 600s kill

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

    # Smooth surrogate for maximizing (d_min/d_max)^2
    # Equivalent to minimizing log(d_max) - log(d_min)
    def objective(x, s=0.1):
        """
        Minimize soft_max(log(d)) - soft_min(log(d))
        which approximates log(d_max) - log(d_min)
        """
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            return 1e10
        
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
            
        dists = pdist(p, metric='euclidean')
        
        # Avoid log(0)
        min_dist = np.min(dists)
        if min_dist < 1e-10:
            return 1e10 + 1.0 / min_dist
            
        log_dists = np.log(dists)
        
        # Soft max of log_dists: log(sum(exp(log_dists/s))/n) * s ?
        # Actually, max(log_dists) = log(max(dists)).
        # We want to approximate max(log_dists) and min(log_dists).
        
        # Softmax approximation for max(y):
        # M = max(y)
        # approx_max = M + s * log(mean(exp((y-M)/s)))
        
        max_log = np.max(log_dists)
        exp_scaled_max = np.exp((log_dists - max_log) / s)
        soft_max_log = max_log + s * np.log(np.mean(exp_scaled_max))
        
        min_log = np.min(log_dists)
        exp_scaled_min = np.exp((min_log - log_dists) / s) # Note: min is -max(-y)
        # Or simply:
        # approx_min = m - s * log(mean(exp((m-y)/s))) where m=min(y)
        soft_min_log = min_log - s * np.log(np.mean(np.exp((min_log - log_dists) / s)))
        
        # We want to minimize (approx_max - approx_min)
        return soft_max_log - soft_min_log

    # Initial configurations
    initializations = []
    
    # 1. Icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
    poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    initializations.append(np.vstack([ico_verts, poles]))
    
    # 2. Icosahedron + 2 points slightly off poles
    offset = 0.1
    poles_off = np.array([[offset, 0, np.sqrt(1-offset**2)], [0, offset, -np.sqrt(1-offset**2)]])
    # Need to ensure they are on sphere or just near
    initializations.append(np.vstack([ico_verts, poles_off]))

    # 3. Fibonacci sphere (golden spiral) - known to give good initial distribution
    def fibonacci_sphere(n_points):
        """Generate points on a sphere using Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    fib_pts = fibonacci_sphere(n)
    initializations.append(fib_pts)
    
    # 4. More Fibonacci with slight perturbations
    np.random.seed(42)
    for _ in range(3):
        perturbed = fib_pts + 0.05 * np.random.randn(n, d)
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1)[:, np.newaxis]
        initializations.append(perturbed)
    
    # 5. Random points on sphere
    for _ in range(3):
        rand_pts = np.random.randn(n, d)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        initializations.append(rand_pts)

    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    # Optimization loop
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > time_limit - 60:
            break
            
        try:
            # Use L-BFGS-B
            # Start with a moderate s, then decrease s for polish
            for s_val in [0.5, 0.2, 0.1, 0.05, 0.02]:
                if time.time() - start_time > time_limit - 20:
                    break
                    
                res = minimize(objective, init_pts.ravel(), method='L-BFGS-B', 
                               args=(s_val,),
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    # Normalize scale? The ratio is scale invariant, but keeping them finite is good.
                    # We can scale them to have max distance = 1 for consistency
                    if np.all(np.isfinite(opt_points)):
                        dists_opt = pdist(opt_points)
                        max_d = np.max(dists_opt)
                        if max_d > 0:
                            opt_points = opt_points / max_d * 1.0 # Scale to max dist 1
                            
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                            init_pts = opt_points.copy() # Use improved point for next s
                        else:
                            # If it didn't improve, stick with previous best for this init branch?
                            # Or try next s with original init? Usually polishing from current is better.
                            pass
                            
        except Exception:
            pass

    # Final exact polish pass: use a very small s for high-precision local optimization
    if time.time() - start_time < time_limit - 30:
        try:
            for s_val in [0.01, 0.005, 0.001]:
                if time.time() - start_time > time_limit - 10:
                    break
                    
                res = minimize(objective, best_points.ravel(), method='L-BFGS-B', 
                               args=(s_val,),
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        dists_opt = pdist(opt_points)
                        max_d = np.max(dists_opt)
                        if max_d > 0:
                            opt_points = opt_points / max_d * 1.0
                            
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

# EVOLVE-BLOCK-END
