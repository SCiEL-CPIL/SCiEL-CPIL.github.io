# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
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
    time_limit = 600  # seconds
    safety_margin = 30  # seconds before limit to stop
    
    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def band_objective(x, gamma=0.01, beta=0.05):
        """
        Smooth objective: normalize distances by soft_max, then penalize
        deviation from the ideal band [1-gamma, 1+gamma].
        This is smoother than the ratio objective near the optimum.
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        if np.all(dists == 0):
            return 1e10

        # Soft max for normalization
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_max <= 1e-12:
            return 1e10

        # Normalized distances
        norm_dists = dists / soft_max
        
        # Penalize distances outside [1-gamma, 1+gamma]
        # Using smooth hinge: max(0, |x - 1| - gamma)^2
        deviations = np.abs(norm_dists - 1.0)
        excess = np.maximum(deviations - gamma, 0.0)
        penalty = np.sum(excess ** 2)
        
        # Also add a small regularization to keep points well-spread
        center = np.mean(p, axis=0)
        spread = np.mean(np.sum((p - center) ** 2, axis=1))
        
        return penalty + 1e-4 * spread

    def t_objective(x):
        """
        Epigraph-style: minimize t such that all distances are in [1/t, t]
        after scaling. We minimize t^2 with smooth constraints.
        x contains 42 point coords + 1 value for t (index 42).
        """
        p = x[:n*d].reshape((n, d))
        t = x[n*d]
        
        if not np.all(np.isfinite(p)) or not np.isfinite(t):
            return 1e10
        if t < 1e-6:
            return 1e10

        dists = pdist(p, metric='euclidean')
        d_max = np.max(dists)
        d_min = np.min(dists)
        
        if d_max <= 1e-12:
            return 1e10

        # Scale so that d_max = 1
        scale = d_max
        norm_dists = dists / scale
        
        # We want all norm_dists >= 1/t (i.e., d_min >= d_max/t)
        # Penalty for violations: smooth max(0, 1/t - norm_dists)^2
        lower_bound = 1.0 / t
        lower_violations = np.maximum(lower_bound - norm_dists, 0.0)
        lower_penalty = np.sum(lower_violations ** 2)
        
        # Also penalize if t is too large (we want to minimize t)
        t_penalty = t ** 2
        
        return t_penalty + 100.0 * lower_penalty

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        x: flat array of length 42
        """
        p = x.reshape((n, d))
        
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def exact_objective(x):
        """Exact (d_max/d_min)^2 for final polish."""
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max == 0:
            return 1e10
        return (d_max / d_min) ** 2

    # Initial guess: Icosahedron vertices + 2 poles, normalized
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    # Add two points to make 14. Poles are a natural choice for symmetry.
    points_init = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    points_init = points_init - np.mean(points_init, axis=0)
    
    best_points = points_init.copy()
    best_score = get_score(best_points)
    
    # 1. Global Optimization using Differential Evolution
    if time.time() - start_time < time_limit - safety_margin:
        try:
            de_deadline = start_time + (time_limit * 0.30)
            
            def de_objective(x):
                if time.time() > de_deadline:
                    return 1e10
                return smooth_objective(x, beta=0.1)
            
            res_de = differential_evolution(
                de_objective, 
                bounds=[(-5, 5)] * (n * d), 
                maxiter=35, 
                popsize=12, 
                tol=1e-5, 
                seed=42,
                init='latinhypercube'
            )
            
            if res_de.fun < 1e9:
                cand_points = res_de.x.reshape((n, d))
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 2. Local Refinement using L-BFGS-B with adaptive beta
    candidates_to_polish = [best_points.copy(), points_init.copy()]
    
    # Add random perturbations
    np.random.seed(123)
    for _ in range(6):
        perturb = best_points + np.random.randn(n, d) * 0.15
        candidates_to_polish.append(perturb)
    
    # Fibonacci sphere initial guesses
    def fibonacci_sphere(n_points, seed=0):
        rng = np.random.RandomState(seed)
        points = []
        for i in range(n_points):
            z = 1 - 2 * (i + 0.5) / n_points
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / n_points
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        pts = pts - np.mean(pts, axis=0)
        return pts
    
    for s in [1, 2, 3]:
        candidates_to_polish.append(fibonacci_sphere(n, seed=s))
    
    for init_pts in candidates_to_polish:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            # First polish with moderate beta
            res_local = minimize(
                lambda x: smooth_objective(x, beta=0.05), 
                init_pts.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < 1e9:
                opt_points = res_local.x.reshape((n, d))
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
                    
                    # Second polish with smaller beta
                    if time.time() - start_time < time_limit - safety_margin:
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

    # 3. Band objective polish - smoother near optimum
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_band = minimize(
                lambda x: band_objective(x, gamma=0.02, beta=0.05),
                best_points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_band.fun < 1e9:
                band_points = res_band.x.reshape((n, d))
                band_score = get_score(band_points)
                if band_score > best_score:
                    best_points = band_points
                    best_score = band_score
        except Exception:
            pass

    # 4. T-objective (epigraph) polish
    if time.time() - start_time < time_limit - safety_margin:
        try:
            # Initial t estimate from current best
            dists_best = pdist(best_points, metric='euclidean')
            t_init = np.max(dists_best) / np.min(dists_best)
            
            x_t_init = np.concatenate([best_points.ravel(), [t_init]])
            
            res_t = minimize(
                t_objective,
                x_t_init,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * (n * d) + [(1.0, 100.0)],
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_t.fun < 1e9:
                t_points = res_t.x[:n*d].reshape((n, d))
                t_score = get_score(t_points)
                if t_score > best_score:
                    best_points = t_points
                    best_score = t_score
        except Exception:
            pass

    # 5. Final high-precision polish with exact objective
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    # 6. Final band polish with very tight gamma
    if time.time() - start_time < time_limit - 15:
        try:
            res_band2 = minimize(
                lambda x: band_objective(x, gamma=0.005, beta=0.02),
                best_points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_band2.fun < 1e9:
                band2_points = res_band2.x.reshape((n, d))
                band2_score = get_score(band2_points)
                if band2_score > best_score:
                    best_points = band2_points
                    best_score = band2_score
        except Exception:
            pass

    # 7. Try a few more random restarts if time allows
    if time.time() - start_time < time_limit - 15:
        for seed in [777, 888, 999, 1000, 1111, 2222, 3333]:
            if time.time() - start_time > time_limit - 10:
                break
            try:
                rp = fibonacci_sphere(n, seed=seed)
                res_q = minimize(
                    lambda x: smooth_objective(x, beta=0.03), 
                    rp.ravel(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                if res_q.fun < 1e9:
                    qp = res_q.x.reshape((n, d))
                    qs = get_score(qp)
                    if qs > best_score:
                        best_points = qp
                        best_score = qs
                        
                        # Quick band polish
                        if time.time() - start_time < time_limit - 10:
                            res_q2 = minimize(
                                lambda x: band_objective(x, gamma=0.01, beta=0.03),
                                qp.ravel(),
                                method='L-BFGS-B',
                                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11}
                            )
                            if res_q2.fun < 1e9:
                                qp2 = res_q2.x.reshape((n, d))
                                qs2 = get_score(qp2)
                                if qs2 > best_score:
                                    best_points = qp2
                                    best_score = qs2
            except Exception:
                pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        return points_init

    return best_points

# EVOLVE-BLOCK-END
