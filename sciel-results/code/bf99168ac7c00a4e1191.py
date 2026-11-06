import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n = 8
    
    def get_points(r1, r2, phi):
        # Square 1: radius r1, angle offset 0
        # Square 2: radius r2, angle offset phi
        # Angles for square: 0, pi/2, pi, 3pi/2
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + phi
        
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([pts1, pts2])

    def log_score_from_params(r1, r2, phi):
        """
        Calculates the log of the score for the configuration defined by r1, r2, phi.
        The score is Delta / n^n.
        We maximize this.
        """
        pts = get_points(r1, r2, phi)
        
        # Calculate pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        
        # Diameter is max distance
        D = np.max(dists)
        if D == 0:
            return -np.inf
            
        # Scale factor to make diameter 2
        scale = 2.0 / D
        
        # The distances in the scaled set are dists * scale
        # We want sum_{i != j} log(dist_scaled_ij) - n log n
        # sum_{i != j} log(d_ij * scale) = sum_{i != j} (log(d_ij) + log(scale))
        # = sum_{i != j} log(d_ij) + n(n-1) log(scale)
        
        # Note: dists includes diagonal (0). We need i != j.
        # Set diagonal to 1 for log calculation, then subtract?
        # Or just mask it.
        mask = ~np.eye(n, dtype=bool)
        log_dists_sum = np.sum(np.log(dists[mask]))
        
        log_scale = np.log(scale)
        
        log_delta_scaled = log_dists_sum + n * (n - 1) * log_scale
        
        log_score = log_delta_scaled - n * np.log(n)
        
        return log_score

    def objective(params):
        """
        Objective function for minimization.
        Minimizes -log_score.
        """
        r1, r2, phi = params
        # r1 is fixed to 1 in the reduced problem, but let's keep it general for DE if needed?
        # No, we decided to fix r1=1.
        # So params are [r2, phi]
        # But the signature might expect 3 if we use the 3D version.
        # Let's stick to 2D: r2, phi.
        
        # Wait, the function definition above takes r1, r2, phi.
        # If we use 2D optimization, we pass r1=1.
        ls = log_score_from_params(1.0, r2, phi)
        if not np.isfinite(ls):
            return 1e10
        return -ls

    # Initial guess: Regular octagon
    # r1 = 1, r2 = 1, phi = pi/4
    # This corresponds to a regular octagon.
    
    # Differential Evolution on 2 variables: r2, phi
    # r2 range: [0.1, 10.0]
    # phi range: [0, pi/2]
    
    bounds = [(0.1, 10.0), (0.0, np.pi/2)]
    
    best_score = -np.inf
    best_params = (1.0, np.pi/4) # r2, phi
    
    # Evaluate initial guess
    init_score = log_score_from_params(1.0, 1.0, np.pi/4)
    if init_score > best_score:
        best_score = init_score
        best_params = (1.0, np.pi/4)
        
    # Run DE
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=20,
                seed=42,
                tol=1e-8,
                polish=False
            )
            
            de_score = -de_result.fun
            if de_score > best_score:
                best_score = de_score
                best_params = de_result.x
        except Exception as e:
            pass

    # Local Refinement with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if res.fun < 0: # Valid score
                res_score = -res.fun
                if res_score > best_score:
                    best_score = res_score
                    best_params = res.x
        except Exception:
            pass
            
    # Final Polish with even tighter tolerance?
    # Nelder-Mead is usually good enough.
    
    # Construct final points
    r2_opt, phi_opt = best_params
    r1_opt = 1.0
    
    points = get_points(r1_opt, r2_opt, phi_opt)
    
    # Scale to diameter 2
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    D = np.max(dists)
    if D > 0:
        points = points * (2.0 / D)
        
    # Return as list of lists
    return [[float(p[0]), float(p[1])] for p in points]
