import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    n = 8
    
    def compute_score(pts):
        """Compute the score Delta/n^n where Delta is product of ordered pair distances
        after scaling to diameter 2."""
        pts = np.array(pts, dtype=float)
        if pts.shape != (n, 2):
            return -np.inf
        
        # Check distinctness
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        
        if np.any(d < 1e-9):
            return -np.inf
        
        diam = np.max(d)
        if diam < 1e-9:
            return -np.inf
        
        # Scale to diameter 2
        pts_scaled = pts * (2.0 / diam)
        
        # Recalculate distances after scaling
        diff_s = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
        dists_s = np.sqrt(np.sum(diff_s**2, axis=2))
        d_s = dists_s[mask]
        
        # Product of all ordered pair distances (i != j)
        # log(Delta) = sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
        log_delta = np.sum(np.log(d_s))
        delta = np.exp(log_delta)
        score = delta / (n ** n)
        
        return score
    
    def cage_points(r1, r2, theta):
        """Generate 8 points as two concentric squares.
        r1, r2: radii of the two squares
        theta: relative rotation angle between the two squares
        Square 1 has vertices at angles 0, pi/2, pi, 3pi/2 with radius r1
        Square 2 has vertices at angles theta, theta+pi/2, theta+pi, theta+3pi/2 with radius r2
        """
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = theta + angles1
        
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([pts1, pts2])
    
    def cage_negative_score(params):
        """Negative score for optimization (we minimize)."""
        r1, r2, theta = params
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
        if r1 > 1000 or r2 > 1000:
            return 1e10
        
        pts = cage_points(r1, r2, theta)
        score = compute_score(pts)
        if score <= 0:
            return 1e10
        return -np.log(score)
    
    # Strategy 1: Optimize the cage parameterization (3 parameters)
    best_score = -np.inf
    best_pts = None
    
    # Grid search over the 3D parameter space
    # r1 and r2 should be positive, theta in [0, pi/2] by symmetry
    # The regular octagon corresponds to r1 = r2 and theta = pi/8 (45 degrees / 2)
    # Let's do a coarse grid first
    
    if time.time() < deadline - 10:
        # Coarse grid search
        r1_vals = np.linspace(0.3, 1.5, 15)
        r2_vals = np.linspace(0.3, 1.5, 15)
        theta_vals = np.linspace(0, np.pi/2, 11)
        
        for r1 in r1_vals:
            if time.time() > deadline - 5:
                break
            for r2 in r2_vals:
                for theta in theta_vals:
                    params = np.array([r1, r2, theta])
                    neg_score = cage_negative_score(params)
                    if neg_score < 1e9:
                        score = np.exp(-neg_score)
                        if score > best_score:
                            best_score = score
                            best_pts = cage_points(r1, r2, theta)
    
    # Local refinement from best grid point and several other starting points
    if time.time() < deadline - 5 and best_pts is not None:
        # Refine the best grid point
        initial_params = np.array([
            cage_points(*[0.5, 0.5, np.pi/8])[0, 0],  # placeholder, we'll re-derive
            0.5, np.pi/8
        ])
        
        # Actually, let's just use several good starting points
        start_params_list = [
            np.array([1.0, 1.0, np.pi/8]),    # Regular octagon
            np.array([1.0, 0.5, np.pi/8]),
            np.array([1.0, 0.7, np.pi/8]),
            np.array([1.0, 0.8, np.pi/8]),
            np.array([1.0, 0.6, np.pi/4]),
            np.array([1.0, 0.4, np.pi/4]),
            np.array([0.8, 1.0, np.pi/8]),
            np.array([1.2, 0.6, np.pi/8]),
            np.array([1.0, 0.5, 0.0]),
            np.array([1.0, 0.5, np.pi/2]),
        ]
        
        # Add the best from grid search
        if best_pts is not None:
            # Try to recover parameters from best_pts is complex, just add a few more
            pass
        
        for x0 in start_params_list:
            if time.time() > deadline - 3:
                break
            
            try:
                res = minimize(cage_negative_score, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if res.fun < 1e9:
                    score = np.exp(-res.fun)
                    if score > best_score:
                        best_score = score
                        best_pts = cage_points(*res.x)
            except Exception:
                pass
    
    # Strategy 2: Full 16-parameter optimization from cage initializations
    if time.time() < deadline - 15:
        # Use the best cage configuration as starting point for full optimization
        if best_pts is None:
            # Fallback: regular octagon
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
            best_score = compute_score(best_pts)
        
        def full_objective(x):
            pts = x.reshape(n, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            mask = ~np.eye(n, dtype=bool)
            d_sq = dists_sq[mask]
            
            if np.any(d_sq < 1e-12):
                return 1e10
            
            # We want to maximize product of distances / diameter^8
            # Equivalent to maximizing sum(log(d_ij)) - 8*log(diam)
            # But diam is max(d_ij), which is non-smooth
            # Alternative: normalize by a reference distance
            # Use the sum of all pair distances as a normalization
            log_sum = 0.5 * np.sum(np.log(d_sq))
            # For 8 points, there are 28 unordered pairs, 56 ordered pairs
            # We want to maximize product, so minimize -log_sum
            # But we need to handle the diameter constraint
            # Let's just use the raw log product and handle scaling separately
            return -log_sum
        
        def full_constraint(x):
            pts = x.reshape(n, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            mask = ~np.eye(n, dtype=bool)
            d_sq = dists_sq[mask]
            # Constraint: d^2 <= 1 (diameter <= 1)
            return 1 - d_sq
        
        constraints = [{'type': 'ineq', 'fun': full_constraint}]
        
        # Scale best_pts to have diameter 1 for the constraint
        diff_b = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists_b = np.sqrt(np.sum(diff_b**2, axis=2))
        mask_b = ~np.eye(n, dtype=bool)
        diam_b = np.max(dists_b[mask_b])
        
        if diam_b > 1e-9:
            x0_full = (best_pts * (1.0 / diam_b)).flatten()
        else:
            x0_full = best_pts.flatten()
        
        try:
            res_full = minimize(full_objective, x0_full, method='SLSQP',
                               constraints=constraints,
                               options={'maxiter': 2000, 'ftol': 1e-12})
            
            if res_full.fun < 1e9:
                pts_full = res_full.x.reshape(n, 2)
                score_full = compute_score(pts_full)
                if score_full > best_score:
                    best_score = score_full
                    best_pts = pts_full
        except Exception:
            pass
    
    # Strategy 3: Try differential evolution on full 16 params if time permits
    if time.time() < deadline - 30 and best_pts is not None:
        try:
            # Bounds: points should be within [-2, 2] roughly
            bounds = [(-2, 2)] * 16
            
            def de_objective(x):
                pts = x.reshape(n, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                mask = ~np.eye(n, dtype=bool)
                d_sq = dists_sq[mask]
                
                if np.any(d_sq < 1e-12):
                    return 1e10
                
                # Penalize if diameter > 1
                max_d_sq = np.max(d_sq)
                if max_d_sq > 1.0:
                    penalty = (max_d_sq - 1.0) * 100
                else:
                    penalty = 0
                
                log_sum = 0.5 * np.sum(np.log(d_sq))
                return -log_sum + penalty
            
            de_res = differential_evolution(de_objective, bounds, maxiter=50,
                                            popsize=10, tol=1e-8,
                                            seed=42, polish=False,
                                            updating='deferred')
            
            if de_res.fun < 1e9:
                pts_de = de_res.x.reshape(n, 2)
                score_de = compute_score(pts_de)
                if score_de > best_score:
                    best_score = score_de
                    best_pts = pts_de
        except Exception:
            pass
    
    # Final fallback
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure coordinates are within bounds
    best_pts = np.clip(best_pts, -1e6, 1e6)
    
    return best_pts.tolist()
