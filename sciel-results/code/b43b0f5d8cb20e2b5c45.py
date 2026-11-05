import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
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

    # Phase 1: Differential Evolution for global exploration
    # Use a smooth surrogate for DE to handle non-differentiability of max
    k_de = 100.0
    
    def smooth_surrogate(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists_sq = np.maximum(dists_sq, 1e-15)
        mask = ~np.eye(n, dtype=bool)
        off_diag_sq = dists_sq[mask]
        
        log_dists = 0.5 * np.log(off_diag_sq)
        sum_log_dists = np.sum(log_dists)
        
        d_k = np.power(off_diag_sq, k_de / 2.0)
        sum_d_k = np.sum(d_k)
        if sum_d_k < 1e-15:
            return 1e10
        log_d_soft = (1.0 / k_de) * np.log(sum_d_k)
        
        return -sum_log_dists + 56.0 * log_d_soft

    bounds = [(-2, 2)] * (2 * n)
    
    # Run DE with limited budget
    de_time_budget = 60.0
    de_start = time.time()
    
    try:
        de_result = differential_evolution(
            smooth_surrogate,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            polish=False,
            update_interval=5,
            callback=None
        )
        if de_result.fun < 1e9:
            candidate = de_result.x.reshape(n, 2)
            best_state = try_update(candidate, best_state)
    except Exception:
        pass
    
    # Phase 2: Multi-start Nelder-Mead on exact objective
    # Generate diverse starting points
    np.random.seed(42)
    start_points = []
    
    # Regular octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    start_points.append(reg_oct.flatten())
    
    # If DE found a good point, use it
    if best_state['points'] is not None:
        start_points.append(best_state['points'].flatten())
    
    # Two concentric squares with various parameters
    for r1 in [0.5, 0.7, 0.9, 1.0]:
        for r2 in [0.5, 0.7, 0.9, 1.0]:
            for theta in [0, np.pi/8, np.pi/4]:
                a1 = 2 * np.pi * np.arange(4) / 4
                a2 = 2 * np.pi * np.arange(4) / 4 + theta
                sq1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
                sq2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
                pts = np.vstack([sq1, sq2])
                start_points.append(pts.flatten())
    
    # Random perturbations of regular octagon
    for _ in range(30):
        base_angles = 2 * np.pi * np.arange(n) / n
        perturbed_angles = base_angles + np.random.uniform(-0.5, 0.5, n)
        radii = 1.0 + np.random.uniform(-0.2, 0.2, n)
        pts = np.column_stack([np.cos(perturbed_angles) * radii, np.sin(perturbed_angles) * radii])
        start_points.append(pts.flatten())
    
    # Random points
    for _ in range(30):
        pts = np.random.uniform(-1.5, 1.5, (n, 2))
        start_points.append(pts.flatten())
    
    # Nelder-Mead refinement on exact objective
    nm_time_budget = deadline - time.time() - 5.0
    nm_per_point = max(0.5, nm_time_budget / len(start_points))
    
    for init in start_points:
        if time.time() > deadline - 3:
            break
        
        try:
            res = minimize(
                exact_negative_log_score,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if res.fun < 1e9:
                candidate = res.x.reshape(n, 2)
                best_state = try_update(candidate, best_state)
        except Exception:
            pass
    
    # Phase 3: Final refinement with higher precision Nelder-Mead
    if best_state['points'] is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                exact_negative_log_score,
                best_state['points'].flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
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
