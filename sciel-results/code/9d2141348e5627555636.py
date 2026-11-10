# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a scale-invariant smooth surrogate (soft-min/soft-max) for optimization.
    Combines Differential Evolution for global search with L-BFGS-B and Nelder-Mead for local refinement.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin

    def get_score(pts):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        This is differentiable and suitable for L-BFGS-B.
        beta controls the smoothness; smaller beta approximates the exact metric better.
        """
        p = x.reshape((n, d))
        
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        
        # Soft min
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        # Handle potential overflow in exp
        if max_a_neg > 700:
            return 1e10
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        # Soft max
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        if max_a_pos > 700:
            return 1e10
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 1e-9 or soft_max <= 1e-9:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def exact_objective(x):
        """Exact (d_max/d_min)^2 for final polish (minimizing this maximizes the score)."""
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 1e-9 or d_max <= 1e-9:
            return 1e10
        return (d_max / d_min) ** 2

    # 1. Initialize points
    # Strategy: Use Icosahedron (12 points) + 2 poles as a strong symmetric starting point.
    # Also prepare a Fibonacci sphere alternative.
    
    def make_icosahedron_plus_poles():
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        norms_ico = np.linalg.norm(ico_verts, axis=1)
        ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
        
        # Add two poles
        poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        points_init = np.vstack([ico_verts_norm, poles])
        
        # Center the points to remove translation invariance
        points_init = points_init - np.mean(points_init, axis=0)
        return points_init

    def make_fibonacci_sphere(n_points):
        # Simple deterministic Fibonacci sphere
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            points[i] = [x, y, z]
        points = points - np.mean(points, axis=0)
        return points

    best_points = make_icosahedron_plus_poles()
    best_score = get_score(best_points)
    
    # 2. Global Optimization using Differential Evolution
    # We use the smooth surrogate to guide DE, as it is smoother than the exact metric.
    # DE is robust to non-smoothness and scale invariance if bounds are reasonable.
    if time.time() < deadline - 150.0:
        try:
            de_deadline = time.time() + 120.0 # Allow ~120s for DE
            
            def de_objective(x):
                if time.time() > de_deadline:
                    return 1e10
                # Use a larger beta for DE to make the landscape smoother and easier to navigate
                return smooth_objective(x, beta=0.1)
            
            # Bounds: The optimal configuration will likely be within a reasonable radius.
            # Since the metric is scale invariant, we can fix the scale by bounding coordinates.
            # Centering is not enforced in DE, so we bound around 0.
            bounds = [(-5, 5)] * (n * d)
            
            res_de = differential_evolution(
                de_objective, 
                bounds=bounds, 
                maxiter=50, 
                popsize=20, 
                tol=1e-4, 
                seed=42,
                init='latinhypercube',
                polish=False,  # We do our own polishing
                callback=None
            )
            
            if res_de.fun < 1e9:
                cand_points = res_de.x.reshape((n, d))
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 3. Local Refinement using L-BFGS-B
    # We try to polish the best point found, and also try the initial symmetric points.
    candidates_to_polish = [best_points.copy(), make_icosahedron_plus_poles(), make_fibonacci_sphere(n)]
    
    # Add some random perturbations of the best point to escape local minima
    np.random.seed(123)
    for _ in range(5):
        perturb = best_points + np.random.randn(n, d) * 0.1
        candidates_to_polish.append(perturb)
        
    for init_pts in candidates_to_polish:
        if time.time() > deadline - 40.0:
            break
            
        try:
            # Stage 1: Polish with smooth objective (beta=0.05)
            res_local = minimize(
                lambda x: smooth_objective(x, beta=0.05), 
                init_pts.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < 1e9:
                opt_points = res_local.x.reshape((n, d))
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
                    
                    # Stage 2: Polish with tighter smooth objective (beta=0.01)
                    if time.time() < deadline - 20.0:
                        res_local2 = minimize(
                            lambda x: smooth_objective(x, beta=0.01), 
                            opt_points.ravel(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if res_local2.fun < 1e9:
                            opt_points2 = res_local2.x.reshape((n, d))
                            opt_score2 = get_score(opt_points2)
                            if opt_score2 > best_score:
                                best_points = opt_points2
                                best_score = opt_score2
        except Exception:
            pass

    # 4. Final high-precision polish with exact objective using Nelder-Mead
    # Nelder-Mead doesn't require gradients and can handle the non-differentiable min/max better in the final stage.
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = make_fibonacci_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
