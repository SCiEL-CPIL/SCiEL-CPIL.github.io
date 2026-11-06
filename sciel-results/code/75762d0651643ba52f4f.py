import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def generate_points(r1, r2, theta):
        """
        Generate 8 points: two concentric squares.
        Square 1: radius r1, rotation 0.
        Square 2: radius r2, rotation theta.
        """
        # Square 1 vertices
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Square 2 vertices
        angles2 = angles1 + theta
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([pts1, pts2])

    def compute_score(pts):
        """
        Compute the score: Delta / 8^8 where Delta is product of ordered pair distances
        after scaling the set to diameter 2.
        """
        # 1. Scale to diameter 2
        d_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        d_max = np.max(d_mat)
        if d_max < 1e-9:
            return 0.0, pts
        
        pts_scaled = pts * (2.0 / d_max)
        
        # 2. Compute product of ordered pair distances
        d_mat2 = np.sqrt(np.sum((pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :])**2, axis=2))
        mask = ~np.eye(8, dtype=bool)
        d_safe = np.where(mask, np.maximum(d_mat2, 1e-15), 1.0)
        log_delta = np.sum(np.log(d_safe) * mask)
        delta = np.exp(log_delta)
        score = delta / (8.0 ** 8)
        return score, pts_scaled

    def refine_params(r1, r2, theta, max_iters=100):
        """
        Local refinement of the 3 parameters (r1, r2, theta) using Nelder-Mead.
        We maximize the log-score (equivalent to maximizing score).
        """
        from scipy.optimize import minimize
        
        def objective(params):
            r1, r2, theta = params
            # Ensure radii are positive
            r1 = abs(r1) + 1e-6
            r2 = abs(r2) + 1e-6
            pts = generate_points(r1, r2, theta)
            score, _ = compute_score(pts)
            # Return negative log score for minimization
            # If score is 0, return large penalty
            if score <= 0:
                return 1e6
            return -np.log(score)

        x0 = [r1, r2, theta]
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'adaptive': True})
            r1_opt, r2_opt, theta_opt = res.x
            r1_opt = abs(r1_opt)
            r2_opt = abs(r2_opt)
            return r1_opt, r2_opt, theta_opt
        except Exception:
            return r1, r2, theta

    best_score_global = 0.0
    best_params_global = None
    best_pts_global = None

    # 1. Initialize with regular octagon (which corresponds to r1=1, r2=1, theta=pi/4 
    #    if we consider two squares, but actually regular octagon is a single ring.
    #    However, two concentric squares with r1=r2 and theta=pi/4 form a regular octagon.
    #    Let's start with that.)
    
    # Regular Octagon check
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    sc_oct, pts_oct_scaled = compute_score(pts_oct)
    best_score_global = sc_oct
    best_pts_global = pts_oct_scaled.copy()
    # The params for regular octagon in our model: r1=1, r2=1, theta=pi/4
    best_params_global = (1.0, 1.0, np.pi/4)

    # 2. Global Grid Search over the 3D parameter space
    # We need to determine the range for r1, r2, theta.
    # Due to scaling invariance, we can fix one radius or normalize.
    # However, the relative ratio r2/r1 and theta are the key variables.
    # Let's fix r1 = 1.0 and search r2 in [0.1, 1.5] and theta in [0, pi/2].
    # Note: theta is periodic with pi/2 due to square symmetry.
    
    # Grid resolution:
    # We have ~145 seconds.
    # Let's try a coarse grid first, then refine the top candidates.
    
    r1_fixed = 1.0
    r2_vals = np.linspace(0.2, 1.2, 20) # 20 values
    theta_vals = np.linspace(0, np.pi/2, 30) # 30 values
    
    # Total candidates: 20 * 30 = 600.
    # Each evaluation is fast (8 points).
    # We can afford a finer grid.
    
    # Let's do a 2-stage approach:
    # Stage 1: Coarse grid
    r2_coarse = np.linspace(0.1, 1.5, 15)
    theta_coarse = np.linspace(0, np.pi/2, 20)
    
    candidates = []
    
    for r2 in r2_coarse:
        for theta in theta_coarse:
            if time.time() > deadline - 10:
                break
            pts = generate_points(r1_fixed, r2, theta)
            score, _ = compute_score(pts)
            candidates.append((score, r1_fixed, r2, theta))
    
    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 20 candidates for refinement
    top_candidates = candidates[:20]
    
    # Also add the regular octagon params explicitly to top list if not there
    # (It should be there if the grid hits it, but let's be safe)
    if best_params_global not in [c[1:] for c in top_candidates]:
         top_candidates.append((best_score_global, *best_params_global))

    # Stage 2: Local refinement of top candidates
    for score, r1, r2, theta in top_candidates:
        if time.time() > deadline - 5:
            break
            
        # Refine using Nelder-Mead
        r1_opt, r2_opt, theta_opt = refine_params(r1, r2, theta)
        
        # Evaluate refined points
        pts_refined = generate_points(r1_opt, r2_opt, theta_opt)
        score_refined, pts_refined_scaled = compute_score(pts_refined)
        
        if score_refined > best_score_global:
            best_score_global = score_refined
            best_pts_global = pts_refined_scaled.copy()
            best_params_global = (r1_opt, r2_opt, theta_opt)
            
            # If we found a significantly better score, try to refine it even more
            # or check neighbors
            if time.time() < deadline - 10:
                 # Do a second pass of refinement with tighter tolerance
                 r1_opt2, r2_opt2, theta_opt2 = refine_params(r1_opt, r2_opt, theta_opt, max_iters=200)
                 pts_refined2 = generate_points(r1_opt2, r2_opt2, theta_opt2)
                 score_refined2, pts_refined2_scaled = compute_score(pts_refined2)
                 if score_refined2 > best_score_global:
                     best_score_global = score_refined2
                     best_pts_global = pts_refined2_scaled.copy()
                     best_params_global = (r1_opt2, r2_opt2, theta_opt2)

    # Fallback: If the grid search didn't find anything better than octagon, 
    # try a few random perturbations of the octagon in the 3D space
    # to ensure we don't miss a nearby optimum if the grid was too coarse.
    if best_score_global <= 1.0001:
        np.random.seed(42)
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            r1 = 1.0 + np.random.normal(0, 0.1)
            r2 = 1.0 + np.random.normal(0, 0.1)
            theta = np.pi/4 + np.random.normal(0, 0.1)
            
            r1_opt, r2_opt, theta_opt = refine_params(r1, r2, theta, max_iters=50)
            pts_refined = generate_points(r1_opt, r2_opt, theta_opt)
            score_refined, pts_refined_scaled = compute_score(pts_refined)
            
            if score_refined > best_score_global:
                best_score_global = score_refined
                best_pts_global = pts_refined_scaled.copy()

    # Final check and return
    if best_pts_global is None:
        # Should not happen, but fallback to regular octagon
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        best_pts_global = np.column_stack([np.cos(angles), np.sin(angles)])
        # Scale to diameter 2
        d_mat = np.sqrt(np.sum((best_pts_global[:, np.newaxis, :] - best_pts_global[np.newaxis, :, :])**2, axis=2))
        d_max = np.max(d_mat)
        if d_max > 1e-9:
            best_pts_global = best_pts_global * (2.0 / d_max)

    # Ensure distinct points and valid coordinates
    best_pts_global = np.clip(best_pts_global, -1e6, 1e6)
    
    return best_pts_global.tolist()

if __name__ == "__main__":
    # Quick self-test
    pts = construct()
    print(f"Score: {compute_score(np.array(pts))[0]}")
    print(f"Points: {pts}")
