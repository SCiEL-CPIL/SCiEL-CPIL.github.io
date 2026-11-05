import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 8
    
    def exact_log_score(points):
        """
        Calculate the exact log-score: 
        log( (prod_{i!=j} d'_ij) / n^n )
        where d'_ij = d_ij * (2 / diameter)
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        
        if len(off_diag_dists) == 0:
            return -np.inf
        
        diameter = np.max(off_diag_dists)
        if diameter < 1e-15:
            return -np.inf
        
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        
        log_prod = np.sum(np.log(scaled_dists))
        log_score = log_prod - n * np.log(n)
        return log_score

    def exact_negative_log_score(x):
        """
        Exact negative log-score for optimization (minimization).
        x is flattened points (16 floats).
        """
        pts = x.reshape(n, 2)
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return 1e10
        s = exact_log_score(pts)
        if not np.isfinite(s):
            return 1e10
        return -s

    best_state = {'score': -np.inf, 'points': None}
    
    def try_update(points, best_state):
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return best_state
        
        # Check distinct points
        d = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-8:
            return best_state
            
        s = exact_log_score(points)
        if np.isfinite(s) and s > best_state['score']:
            return {'score': s, 'points': points.copy()}
        return best_state

    # Phase 1: Search in the "Two Concentric Squares" family
    # This is a 3D search space (r1, r2, theta) which is much smaller than 16D.
    # The structure is:
    # Square 1: angles 0, pi/2, pi, 3pi/2, radius r1
    # Square 2: angles theta, theta+pi/2, theta+pi, theta+3pi/2, radius r2
    
    def two_squares_params_to_points(params):
        r1, r2, theta = params
        r1 = abs(r1)
        r2 = abs(r2)
        
        a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        a2 = a1 + theta
        
        pts1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        pts2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        
        return np.vstack([pts1, pts2])

    def two_squares_objective(params):
        pts = two_squares_params_to_points(params)
        s = exact_log_score(pts)
        if not np.isfinite(s):
            return 1e10
        return -s

    # Bounds for r1, r2, theta
    # Scale is arbitrary, so we can restrict r1, r2 to a range.
    # theta is periodic with pi/2.
    bounds_ts = [(0.1, 10.0), (0.1, 10.0), (0.0, np.pi/2)]
    
    try:
        de_ts = differential_evolution(
            two_squares_objective,
            bounds_ts,
            seed=42,
            maxiter=300,
            popsize=20,
            tol=1e-10,
            polish=True,
            polish_options={'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}}
        )
        if de_ts.fun < 1e9:
            candidate_pts = two_squares_params_to_points(de_ts.x)
            best_state = try_update(candidate_pts, best_state)
    except Exception:
        pass

    # Phase 2: Global Search in 16D using Smooth Surrogate
    # The exact objective is non-smooth due to max(). 
    # We use a soft-max approximation for DE.
    
    k_de = 50.0
    
    def smooth_surrogate(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists_sq = np.maximum(dists_sq, 1e-15)
        mask = ~np.eye(n, dtype=bool)
        off_diag_sq = dists_sq[mask]
        
        # Log product of distances: sum log(d_ij) = 0.5 sum log(d_ij^2)
        sum_log_dists = 0.5 * np.sum(np.log(off_diag_sq))
        
        # Soft max of distances: (1/k) log(sum(d^k)) approximates max(d)
        # We want to maximize sum_log_dists - 56 * log(diameter)
        # So we minimize -sum_log_dists + 56 * log(diameter)
        
        d_k = np.power(off_diag_sq, k_de / 2.0)
        sum_d_k = np.sum(d_k)
        if sum_d_k < 1e-15:
            return 1e10
        log_d_soft = (1.0 / k_de) * np.log(sum_d_k)
        
        return -sum_log_dists + 56.0 * log_d_soft

    bounds = [(-2, 2)] * (2 * n)
    
    # Run DE with limited budget
    de_start = time.time()
    try:
        de_result = differential_evolution(
            smooth_surrogate,
            bounds,
            seed=123,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            polish=False,
            update_interval=5
        )
        if de_result.fun < 1e9:
            candidate = de_result.x.reshape(n, 2)
            best_state = try_update(candidate, best_state)
    except Exception:
        pass

    # Phase 3: Multi-start Nelder-Mead on Exact Objective
    # We take the best points from Phase 1 and 2, plus some random starts, 
    # and refine them using the exact objective.
    
    start_points = []
    
    # Regular octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    start_points.append(reg_oct.flatten())
    
    # Best from Two Squares
    if best_state['points'] is not None:
        start_points.append(best_state['points'].flatten())
    
    # Add some random perturbations of the current best
    if best_state['points'] is not None:
        for _ in range(5):
            perturb = np.random.normal(0, 0.1, (n, 2))
            start_points.append((best_state['points'] + perturb).flatten())
    
    # Add random points
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.uniform(-1.5, 1.5, (n, 2))
        start_points.append(pts.flatten())
    
    nm_deadline = deadline - 5.0
    for init in start_points:
        if time.time() > nm_deadline:
            break
        try:
            res = minimize(
                exact_negative_log_score,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if res.fun < 1e9:
                candidate = res.x.reshape(n, 2)
                best_state = try_update(candidate, best_state)
        except Exception:
            pass

    # Final check and return
    if best_state['points'] is None:
        best_state = {'score': 0, 'points': reg_oct}
        
    pts = best_state['points']
    
    # Validation
    dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        pts = reg_oct

    max_coord = np.max(np.abs(pts))
    if max_coord > 1e6:
        pts = pts / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in pts]
