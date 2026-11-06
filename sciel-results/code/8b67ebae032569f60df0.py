import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 8
    
    def compute_score(pts):
        """Compute Delta/n^n where Delta is product of ordered pair distances
        after scaling to diameter 2. Returns -inf for invalid inputs."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, 2):
            return -np.inf
        
        # Check distinctness
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        d_sq = dists_sq[mask]
        
        if np.any(d_sq < 1e-12):
            return -np.inf
        
        diam_sq = np.max(d_sq)
        if diam_sq < 1e-12:
            return -np.inf
        
        # Scale to diameter 2: multiply by 2/sqrt(diam_sq)
        scale = 2.0 / np.sqrt(diam_sq)
        pts_scaled = pts * scale
        
        # Recalculate squared distances after scaling
        diff_s = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
        dists_sq_s = np.sum(diff_s**2, axis=2)
        d_sq_s = dists_sq_s[mask]
        
        # Product of all ordered pair distances
        # log(Delta) = sum_{i!=j} log(d_ij) = sum_{i!=j} 0.5*log(d_ij^2)
        log_delta = 0.5 * np.sum(np.log(d_sq_s))
        score = np.exp(log_delta) / (n ** n)
        
        return score
    
    def neg_log_score(pts):
        """Negative log of the score, for minimization. Returns large value for invalid."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, 2):
            return 1e10
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        d_sq = dists_sq[mask]
        
        if np.any(d_sq < 1e-12):
            return 1e10
        
        diam_sq = np.max(d_sq)
        if diam_sq < 1e-12:
            return 1e10
        
        # log(Delta/n^n) = 0.5*sum(log(d_sq)) - n*log(n) - n*log(diam_sq/4)
        # Wait, let me redo this carefully.
        # After scaling to diameter 2, each distance d_ij becomes d_ij * 2/sqrt(diam_sq)
        # So log(d_ij_scaled) = log(d_ij) + log(2) - 0.5*log(diam_sq)
        # log(Delta) = sum_{i!=j} [log(d_ij) + log(2) - 0.5*log(diam_sq)]
        # = sum_{i!=j} log(d_ij) + 56*log(2) - 56*0.5*log(diam_sq)
        # = sum_{i!=j} log(d_ij) + 56*log(2) - 28*log(diam_sq)
        # score = Delta / n^n
        # log(score) = log(Delta) - n*log(n)
        # = sum_{i!=j} log(d_ij) + 56*log(2) - 28*log(diam_sq) - n*log(n)
        # 
        # For minimization, we want to minimize -log(score)
        # = -sum_{i!=j} log(d_ij) - 56*log(2) + 28*log(diam_sq) + n*log(n)
        # The constants don't matter for optimization, so we can just minimize:
        # -sum_{i!=j} log(d_ij) + 28*log(diam_sq)
        # = -0.5*sum_{i!=j} log(d_ij^2) + 28*log(diam_sq)
        
        log_sum_d_sq = np.sum(np.log(d_sq))
        # -0.5 * log_sum_d_sq + 28 * log(diam_sq)
        val = -0.5 * log_sum_d_sq + 28.0 * np.log(diam_sq)
        return val
    
    # Generate initial configurations
    def regular_octagon():
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def cage_config(r1, r2, theta):
        """Two concentric squares with radii r1, r2 and relative rotation theta."""
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = theta + angles1
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])
    
    # Start with regular octagon
    best_pts = regular_octagon()
    best_score = compute_score(best_pts)
    best_neg_log = neg_log_score(best_pts)
    
    # Try several cage configurations
    cage_starts = [
        cage_config(1.0, 1.0, np.pi/8),      # regular octagon
        cage_config(1.0, 0.7, np.pi/8),
        cage_config(1.0, 0.6, np.pi/8),
        cage_config(1.0, 0.8, np.pi/8),
        cage_config(1.2, 0.6, np.pi/8),
        cage_config(0.8, 1.2, np.pi/8),
        cage_config(1.0, 0.5, np.pi/6),
        cage_config(1.0, 0.5, np.pi/4),
        cage_config(1.1, 0.7, np.pi/8),
        cage_config(0.9, 1.1, np.pi/8),
    ]
    
    for cfg in cage_starts:
        if time.time() > deadline - 5:
            break
        s = compute_score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()
            best_neg_log = neg_log_score(best_pts)
    
    # Local optimization using neg_log_score (avoids overflow/underflow)
    def polish(pts0, maxiter=5000, ftol=1e-14):
        """Polish a configuration using L-BFGS-B."""
        x0 = pts0.flatten()
        
        def obj(x):
            pts = x.reshape(n, 2)
            return neg_log_score(pts)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-10})
            if res.fun < 1e9:
                return res.x.reshape(n, 2)
        except Exception:
            pass
        return pts0
    
    # Polish from best configuration
    if time.time() < deadline - 10:
        polished = polish(best_pts)
        s = compute_score(polished)
        if s > best_score:
            best_score = s
            best_pts = polished
            best_neg_log = neg_log_score(best_pts)
    
    # Polish from each cage configuration
    if time.time() < deadline - 15:
        for cfg in cage_starts:
            if time.time() > deadline - 5:
                break
            polished = polish(cfg, maxiter=3000)
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_pts = polished
                best_neg_log = neg_log_score(best_pts)
    
    # Differential evolution for global search
    if time.time() < deadline - 30:
        def de_obj(x):
            pts = x.reshape(n, 2)
            return neg_log_score(pts)
        
        # Bounds: keep points within a reasonable range
        # The diameter will be normalized, so we can bound coordinates
        bounds = [(-3, 3)] * (2 * n)
        
        try:
            de_res = differential_evolution(de_obj, bounds, maxiter=80,
                                            popsize=15, tol=1e-8,
                                            seed=42, polish=False,
                                            updating='deferred',
                                            mutation=(0.5, 1.5),
                                            recombination=0.7)
            
            if de_res.fun < 1e9:
                pts_de = de_res.x.reshape(n, 2)
                s = compute_score(pts_de)
                if s > best_score:
                    best_score = s
                    best_pts = pts_de.copy()
                    best_neg_log = neg_log_score(best_pts)
        except Exception:
            pass
    
    # Final polish from DE result and best
    if time.time() < deadline - 10:
        polished = polish(best_pts, maxiter=10000, ftol=1e-15)
        s = compute_score(polished)
        if s > best_score:
            best_score = s
            best_pts = polished
            best_neg_log = neg_log_score(best_pts)
    
    # One more SLSQP polish for extra precision
    if time.time() < deadline - 5:
        x0 = best_pts.flatten()
        try:
            res = minimize(neg_log_score, x0, method='SLSQP',
                          options={'maxiter': 5000, 'ftol': 1e-15})
            if res.fun < 1e9:
                pts_slsqp = res.x.reshape(n, 2)
                s = compute_score(pts_slsqp)
                if s > best_score:
                    best_score = s
                    best_pts = pts_slsqp
                    best_neg_log = neg_log_score(best_pts)
        except Exception:
            pass
    
    # Final check and return
    if best_pts is None:
        best_pts = regular_octagon()
    
    # Ensure coordinates are within bounds
    best_pts = np.clip(best_pts, -1e6, 1e6)
    
    # Final score check
    final_score = compute_score(best_pts)
    if final_score <= 0 or not np.isfinite(final_score):
        # Fallback to regular octagon
        best_pts = regular_octagon()
    
    return best_pts.tolist()
