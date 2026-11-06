import numpy as np
from scipy.optimize import minimize
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 8

    def compute_score(points):
        """
        Compute Delta/n^n where Delta is product of all ordered pair distances
        after scaling to diameter 2.
        """
        pts = np.asarray(points, dtype=float)
        if len(pts) != n:
            return 0.0
        
        # Check for distinct points
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    return 0.0
        
        # Compute diameter
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return 0.0
        
        # Scale to diameter 2
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        # Compute log of product of all ordered pair distances
        # Ordered pairs means (i,j) and (j,i) both count, so it's 2 * sum over i<j of log(d_ij)
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return 0.0
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        
        # Delta = exp(log_prod_ordered), score = Delta / n^n
        # log(score) = log_prod_ordered - n*log(n)
        log_score = log_prod_ordered - n * math.log(n)
        
        return math.exp(log_score)

    def negative_log_score(points):
        """Objective for minimization: -log(score)"""
        pts = np.asarray(points, dtype=float)
        if len(pts) != n:
            return 1e10
        
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    return 1e10
        
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return 1e10
        
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return 1e10
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        log_score = log_prod_ordered - n * math.log(n)
        
        return -log_score

    # 1. Baseline: Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = compute_score(reg_pts)
    best_pts = reg_pts.copy()

    # 2. Geometric Parameterization (Inspiration 1)
    # x_k = (1 + dr_k) * cos(pi*k/4 + dtheta_k)
    # y_k = (1 + dr_k) * sin(pi*k/4 + dtheta_k)
    
    base_angles = np.arange(n) * (np.pi / n)

    def params_to_points(params):
        dr = params[:n]
        dtheta = params[n:]
        angles = base_angles + dtheta
        radii = 1.0 + dr
        # Ensure radii are positive to avoid collapse
        radii = np.maximum(radii, 1e-6)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    def neg_log_score_from_params(params):
        pts = params_to_points(params)
        # We can use the standard negative_log_score but it takes flat array.
        # Let's adapt it to take points directly or flatten.
        return negative_log_score(pts.flatten())

    # 3. Optimization Strategy
    # Combine the structured parameterization with multi-start Nelder-Mead (Inspiration 1)
    # and potentially a final full-space polish (Current Program).

    best_params = np.zeros(2*n)
    best_param_score = compute_score(params_to_points(best_params))

    rng = np.random.RandomState(42)

    # Define bounds for dr and dtheta
    # dr: radius variation. Allow significant variation.
    # dtheta: angle variation. Allow variation to break symmetry.
    bounds_dr = [(-0.9, 5.0)] * n
    bounds_dtheta = [(-np.pi/2, np.pi/2)] * n

    # Generate starts
    starts = [np.zeros(2*n)] # Regular octagon
    
    # Structured starts
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(5):
            start = np.zeros(2*n)
            start[:n] = rng.uniform(-scale, scale, n)
            start[n:] = rng.uniform(-scale*np.pi/4, scale*np.pi/4, n)
            starts.append(start)
            
    # Random starts
    for _ in range(30):
        start = np.array([rng.uniform(b[0], b[1]) for b in bounds_dr + bounds_dtheta])
        starts.append(start)

    rng.shuffle(starts)

    for i, start in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                neg_log_score_from_params, 
                start, 
                method='Nelder-Mead',
                options={
                    'maxiter': 3000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'adaptive': True
                }
            )
            
            if np.all(np.isfinite(res.x)):
                pts = params_to_points(res.x)
                s = compute_score(pts)
                if s > best_param_score:
                    best_param_score = s
                    best_params = res.x.copy()
                    best_pts = pts.copy()
                    best_score = s

        except Exception:
            pass

    # 4. Final High-Precision Polish
    # Polish the best found parameters in the geometric space
    if time.time() < deadline - 5.0 and best_params is not None:
        try:
            res_polish = minimize(
                neg_log_score_from_params, 
                best_params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 5000, 
                    'xatol': 1e-12, 
                    'fatol': 1e-14, 
                    'adaptive': True
                }
            )
            if np.all(np.isfinite(res_polish.x)):
                pts = params_to_points(res_polish.x)
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    # 5. Optional: Full Space Refinement from Best Candidate
    # If we have time, try to move points in Cartesian space from the best geometric candidate.
    # This allows breaking the radial/angular constraint if the optimum is not well-represented there.
    if time.time() < deadline - 15.0:
        x0 = best_pts.flatten()
        
        # Add a tiny bit of noise to escape potential flat spots
        for noise_level in [0.0, 0.01]:
            if time.time() > deadline - 5.0:
                break
            x_init = x0 + np.random.normal(0, noise_level, size=16)
            try:
                res = minimize(negative_log_score, x_init, method='L-BFGS-B',
                               bounds=[(-10.0, 10.0)] * 16,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if np.all(np.isfinite(res.x)):
                    pts = res.x.reshape(n, 2)
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
            except Exception:
                pass

    # 6. Final Validation and Formatting
    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts
        best_score = compute_score(reg_pts)
        
    # Check distinctness
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-6:
        best_pts = reg_pts
        best_score = compute_score(reg_pts)

    # Ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
