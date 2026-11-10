# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580

    def exact_score_from_pts(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def exact_objective(x):
        pts = x.reshape(n, d)
        return -exact_score_from_pts(pts)

    def smooth_objective(x, beta=100.0):
        """Smooth approximation using log-sum-exp for min and max."""
        pts = x.reshape(n, d)
        dists = pdist(pts)
        # Soft min: -log(sum(exp(-beta*d))) / beta
        # Soft max: log(sum(exp(beta*d))) / beta
        # Use stable versions
        neg_d = -beta * dists
        log_sum_exp_neg = np.max(neg_d) + np.log(np.sum(np.exp(neg_d - np.max(neg_d))))
        soft_min = -log_sum_exp_neg / beta
        
        pos_d = beta * dists
        log_sum_exp_pos = np.max(pos_d) + np.log(np.sum(np.exp(pos_d - np.max(pos_d))))
        soft_max = log_sum_exp_pos / beta
        
        if soft_max < 1e-12:
            return 0.0
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    best_x = None
    best_score = 0.0

    def update_best(x, score):
        nonlocal best_x, best_score
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # Generate 19-point hexagonal cluster
    def gen_hex_cluster(s=1.0):
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1 (radius s)
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([s * np.cos(angle), s * np.sin(angle)])
        # Ring 2 (radius 2s)
        for k in range(12):
            angle = k * np.pi / 6
            pts.append([2 * s * np.cos(angle), 2 * s * np.sin(angle)])
        return np.array(pts)

    cluster_19 = gen_hex_cluster(1.0)

    # Define removal patterns: remove 3 points from the 19
    removal_patterns = [
        [7, 11, 15],   # 0, 120, 240
        [8, 12, 16],   # 30, 150, 270
        [9, 13, 17],   # 60, 180, 300
        [10, 14, 18],  # 90, 210, 330
        [7, 10, 13],   # 0, 90, 180
        [7, 9, 11],    # 0, 60, 120
        [7, 8, 13],    # 0, 30, 180
        [7, 11, 14],   # 0, 120, 210
    ]

    initial_guesses = []
    for rem in removal_patterns:
        mask = np.ones(19, dtype=bool)
        mask[rem] = False
        pts = cluster_19[mask]
        pts = pts - np.mean(pts, axis=0)
        initial_guesses.append(pts)

    # Add random perturbations
    base_pts = initial_guesses[0].copy()
    for _ in range(6):
        perturbed = base_pts + np.random.normal(0, 0.03, (16, 2))
        initial_guesses.append(perturbed)

    # Phase 1: L-BFGS-B with smooth objective on each initial guess
    for i, init_pts in enumerate(initial_guesses):
        if time.time() > deadline - 120:
            break
        try:
            # Use smooth objective with moderate beta
            res = minimize(
                lambda x: smooth_objective(x, beta=50.0),
                init_pts.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts = res.x.reshape(n, d)
            score = exact_score_from_pts(pts)
            update_best(res.x, score)
            
            # Polish with exact objective using Nelder-Mead
            if time.time() < deadline - 60:
                res2 = minimize(
                    exact_objective,
                    res.x,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts2 = res2.x.reshape(n, d)
                score2 = exact_score_from_pts(pts2)
                update_best(res2.x, score2)
        except Exception:
            pass

    # Phase 2: DE with wider bounds
    if best_x is not None and time.time() < deadline - 150:
        try:
            best_pts = best_x.reshape(n, d)
            center = np.mean(best_pts, axis=0)
            spread = np.max(np.abs(best_pts - center)) + 1.0
            bounds = [(center[i] - spread, center[i] + spread) for i in range(n * d)]
            
            res_de = differential_evolution(
                exact_objective,
                bounds,
                maxiter=200,
                popsize=40,
                seed=42,
                tol=1e-10,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            pts = res_de.x.reshape(n, d)
            score = exact_score_from_pts(pts)
            update_best(res_de.x, score)
            
            # Polish DE result
            if time.time() < deadline - 60:
                res2 = minimize(
                    exact_objective,
                    res_de.x,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts2 = res2.x.reshape(n, d)
                score2 = exact_score_from_pts(pts2)
                update_best(res2.x, score2)
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if best_x is not None and time.time() < deadline - 30:
        try:
            res = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            pts = res.x.reshape(n, d)
            score = exact_score_from_pts(pts)
            update_best(res.x, score)
        except Exception:
            pass

    # Fallback
    if best_x is None:
        best_x = initial_guesses[0].ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
