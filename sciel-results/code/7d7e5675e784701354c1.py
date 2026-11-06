import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 8
    
    def compute_score(pts):
        """
        Compute the score Delta/n^n.
        Delta is product of ordered pair distances after scaling to diameter 2.
        Score is scale-invariant if defined as (Prod d_ij) / (Diam^n)^n ? 
        Let's stick to the definition:
        1. Scale pts so max distance is 2.
        2. Compute product of all 56 ordered distances.
        3. Divide by 8^8.
        
        Equivalent to maximizing:
        (Prod d_ij) / (Diam^56) * Constant
        """
        pts = np.array(pts, dtype=float)
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
            
        # We want to maximize log(Delta) - 56 * log(Diam)
        # Delta_scaled = Delta_original * (2/Diam)^56
        # log(Delta_scaled) = log(Delta_original) + 56*(log(2) - log(Diam))
        # Score = exp(log(Delta_scaled)) / 8^8
        # Maximize log(Score) = log(Delta_original) - 56*log(Diam) + Constant
        
        # log(Delta_original) = sum(log(d_ij)) = 0.5 * sum(log(d_ij^2))
        log_d_sq_sum = np.sum(np.log(d_sq))
        log_diam = 0.5 * np.log(diam_sq)
        
        # log_score_unscaled = 0.5 * log_d_sq_sum - 56 * log_diam
        # The constant factors (2^56 / 8^8) don't affect the argmax.
        return 0.5 * log_d_sq_sum - 56.0 * log_diam

    def cage_points(r1, r2, theta):
        """
        Generate 8 points: two concentric squares.
        Square 1: radius r1, angles 0, pi/2, pi, 3pi/2
        Square 2: radius r2, angles theta, theta+pi/2, ...
        """
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = theta + angles1
        
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([pts1, pts2])

    def cage_objective(params):
        r1, r2, theta = params
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
        if r1 > 1000 or r2 > 1000:
            return 1e10
            
        pts = cage_points(r1, r2, theta)
        score = compute_score(pts)
        if score <= -1e9:
            return 1e10
        return -score

    best_score = -np.inf
    best_pts = None
    
    # Strategy 1: Optimize the 3-parameter cage model
    # Regular octagon: r1=1, r2=1, theta=pi/8
    # Try a range of starting points
    
    start_params_list = [
        np.array([1.0, 1.0, np.pi/8]),    # Regular octagon
        np.array([1.0, 0.8, np.pi/8]),
        np.array([1.0, 1.2, np.pi/8]),
        np.array([1.0, 0.5, np.pi/8]),
        np.array([1.0, 0.5, np.pi/4]),
        np.array([1.0, 0.8, np.pi/4]),
        np.array([0.8, 1.0, np.pi/8]),
        np.array([1.2, 1.0, np.pi/8]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, np.pi/4]),
    ]
    
    # Grid search to find a good region
    if time.time() < deadline - 20:
        r1_vals = np.linspace(0.2, 1.5, 10)
        r2_vals = np.linspace(0.2, 1.5, 10)
        theta_vals = np.linspace(0, np.pi/2, 10)
        
        for r1 in r1_vals:
            if time.time() > deadline - 10:
                break
            for r2 in r2_vals:
                for theta in theta_vals:
                    params = np.array([r1, r2, theta])
                    obj = cage_objective(params)
                    if obj < 1e9:
                        score = -obj
                        if score > best_score:
                            best_score = score
                            best_pts = cage_points(r1, r2, theta)

    # Local refinement
    if time.time() < deadline - 10:
        for x0 in start_params_list:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(cage_objective, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res.fun < 1e9:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_pts = cage_points(*res.x)
            except Exception:
                pass

    # Fallback if cage search failed or didn't find anything better than 1.0
    # (Though 1.0 is the regular octagon, which should be found)
    if best_pts is None or best_score < 0.9:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_pts)

    # Strategy 2: Full 16-parameter optimization
    # This can break the symmetry of the cage if the optimal solution is not a cage.
    if time.time() < deadline - 15:
        # Scale best_pts to have diameter 1 for easier constraint handling
        diff_b = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists_b_sq = np.sum(diff_b**2, axis=2)
        mask_b = ~np.eye(n, dtype=bool)
        diam_b_sq = np.max(dists_b_sq[mask_b])
        
        if diam_b_sq > 1e-12:
            x0_full = (best_pts * (1.0 / np.sqrt(diam_b_sq))).flatten()
        else:
            x0_full = best_pts.flatten()

        def full_objective(x):
            pts = x.reshape(n, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            mask = ~np.eye(n, dtype=bool)
            d_sq = dists_sq[mask]
            
            if np.any(d_sq < 1e-12):
                return 1e10
                
            # We want to maximize log(Prod d) - 56 log(Diam)
            # But we are working with points scaled to diam <= 1?
            # No, let's just use the unscaled log score formula.
            # It is scale invariant.
            log_d_sq_sum = np.sum(np.log(d_sq))
            diam_sq = np.max(d_sq)
            log_diam = 0.5 * np.log(diam_sq)
            
            return -(0.5 * log_d_sq_sum - 56.0 * log_diam)

        try:
            # Use L-BFGS-B or SLSQP
            # SLSQP is slower but handles non-smoothness (max) slightly better with finite diffs?
            # Nelder-Mead is robust. Let's try Nelder-Mead on 16 params.
            res_full = minimize(full_objective, x0_full, method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_full.fun < 1e9:
                score_full = -res_full.fun
                if score_full > best_score:
                    best_score = score_full
                    best_pts = res_full.x.reshape(n, 2)
        except Exception:
            pass

    # Strategy 3: Differential Evolution on full 16 params (if time permits)
    # This is expensive but finds global optima.
    if time.time() < deadline - 30:
        try:
            bounds = [(-2, 2)] * 16
            
            def de_objective(x):
                pts = x.reshape(n, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                mask = ~np.eye(n, dtype=bool)
                d_sq = dists_sq[mask]
                
                if np.any(d_sq < 1e-12):
                    return 1e10
                
                log_d_sq_sum = np.sum(np.log(d_sq))
                diam_sq = np.max(d_sq)
                log_diam = 0.5 * np.log(diam_sq)
                
                return -(0.5 * log_d_sq_sum - 56.0 * log_diam)

            de_res = differential_evolution(de_objective, bounds, maxiter=30,
                                            popsize=15, tol=1e-8,
                                            seed=42, polish=False,
                                            updating='deferred')
            
            if de_res.fun < 1e9:
                score_de = -de_res.fun
                if score_de > best_score:
                    best_score = score_de
                    best_pts = de_res.x.reshape(n, 2)
        except Exception:
            pass

    # Final check
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure coordinates are within bounds
    best_pts = np.clip(best_pts, -1e6, 1e6)
    
    return best_pts.tolist()
