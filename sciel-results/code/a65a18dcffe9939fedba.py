import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time
from itertools import combinations

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 590  # Safety margin before 600s kill

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
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
    poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    initializations.append(np.vstack([ico_verts, poles]))
    
    # 2. Icosahedron + 2 points slightly off poles
    offset = 0.1
    poles_off = np.array([[offset, 0, np.sqrt(1-offset**2)], [0, offset, -np.sqrt(1-offset**2)]])
    initializations.append(np.vstack([ico_verts, poles_off]))
    
    # 3. Fibonacci sphere (golden spiral)
    def fibonacci_sphere(n_points):
        """Generate points on a sphere using Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    fib_pts = fibonacci_sphere(n)
    initializations.append(fib_pts)
    
    # 4. Fibonacci with various perturbations
    np.random.seed(42)
    for amp in [0.03, 0.08, 0.15]:
        for _ in range(2):
            perturbed = fib_pts + amp * np.random.randn(n, d)
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1)[:, np.newaxis]
            initializations.append(perturbed)
    
    # 5. Random points on sphere
    for _ in range(4):
        rand_pts = np.random.randn(n, d)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        initializations.append(rand_pts)
    
    # 6. Spherical code: try to place points with good angular separation
    # Start with tetrahedron-like structure and add points
    tetra = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1)[:, np.newaxis]
    # Add 10 more points via rejection sampling
    extra_pts = []
    for _ in range(200):
        cand = np.random.randn(d)
        cand = cand / np.linalg.norm(cand)
        # Check min angle to existing
        existing = np.vstack([tetra, np.array(extra_pts)]) if extra_pts else tetra
        dots = existing @ cand
        if np.max(np.abs(dots)) < 0.7:  # >45 degrees from all
            extra_pts.append(cand)
            if len(extra_pts) == 10:
                break
    if len(extra_pts) == 10:
        initializations.append(np.vstack([tetra, np.array(extra_pts)]))

    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    # Pre-compute scores for all initializations
    init_scores = []
    for init_pts in initializations:
        init_scores.append(get_score(init_pts))
    
    # Sort initializations by score (best first)
    sorted_indices = np.argsort(init_scores)[::-1]
    
    # Optimization loop
    for idx in sorted_indices:
        if time.time() - start_time > time_limit - 45:
            break
        init_pts = initializations[idx].copy()
        
        try:
            # Annealing schedule: coarse to fine
            s_schedule = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01]
            for s_val in s_schedule:
                if time.time() - start_time > time_limit - 15:
                    break
                    
                max_iter = 200 if s_val > 0.1 else 400
                res = minimize(objective, init_pts.ravel(), method='L-BFGS-B', 
                               args=(s_val,),
                               options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        dists_opt = pdist(opt_points)
                        max_d = np.max(dists_opt)
                        if max_d > 0:
                            opt_points = opt_points / max_d
                            
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                        init_pts = opt_points.copy()
                        
        except Exception:
            pass

    # Final exact polish pass: use a very small s for high-precision local optimization
    if time.time() - start_time < time_limit - 25:
        try:
            for s_val in [0.005, 0.002, 0.001, 0.0005]:
                if time.time() - start_time > time_limit - 8:
                    break
                    
                res = minimize(objective, best_points.ravel(), method='L-BFGS-B', 
                               args=(s_val,),
                               options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-15})
                
                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        dists_opt = pdist(opt_points)
                        max_d = np.max(dists_opt)
                        if max_d > 0:
                            opt_points = opt_points / max_d
                            
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                            
        except Exception:
            pass
    
    # Direct local search: identify critical distances and do targeted refinement
    if time.time() - start_time < time_limit - 10:
        try:
            # Get current distances
            dists = pdist(best_points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            # Find indices of min and max distances
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            # Convert flat index to pair indices
            def flat_to_pair(idx, n):
                # pdist uses condensed form: (0,1),(0,2),...,(0,n-1),(1,2),...
                count = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if count == idx:
                            return i, j
                        count += 1
                return 0, 1
            
            min_i, min_j = flat_to_pair(min_idx, n)
            max_i, max_j = flat_to_pair(max_idx, n)
            
            # Try small perturbations of the critical points
            for trial in range(20):
                if time.time() - start_time > time_limit - 5:
                    break
                perturbed = best_points.copy()
                noise_scale = 0.001 * (1 + trial * 0.1)
                # Perturb the points involved in min and max distances
                for pt_idx in [min_i, min_j, max_i, max_j]:
                    perturbed[pt_idx] += noise_scale * np.random.randn(d)
                # Normalize scale
                pert_dists = pdist(perturbed)
                pert_max = np.max(pert_dists)
                if pert_max > 0:
                    perturbed = perturbed / pert_max
                pert_score = get_score(perturbed)
                if pert_score > best_score:
                    best_score = pert_score
                    best_points = perturbed.copy()
                    
        except Exception:
            pass

    # Final safety check
    if not np.all(np.isfinite(best_points)):
        best_points = initializations[0].copy()
        
    return best_points

# EVOLVE-BLOCK-END
