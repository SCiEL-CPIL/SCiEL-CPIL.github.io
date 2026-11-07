"""Root finder for 3 fixed 2-D systems: Adaptive Radial/Log Search + Broyden's Method."""
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Global Search: Adaptive radial search. Start at x0, then expand outward in 
       log-spaced radii with increasing angular resolution. This is more efficient 
       than a fixed grid as it concentrates evaluations where the function value 
       is decreasing (likely near the root) and adapts to the scale of the problem.
    2. Local Refinement: Broyden's method (quasi-Newton) starting from the best 
       found point.
    
    Budget:
    - Time limit: 120s (using 115s internal deadline).
    - Call limit: < 500 calls.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    def time_ok():
        return time.time() - start_time < time_limit

    def safe_eval_F(x):
        """Evaluate F(x) safely, returning (value, norm, success)."""
        try:
            val = F(np.asarray(x, dtype=float))
            val = np.asarray(val, dtype=float)
            if val.shape != (2,) or not np.all(np.isfinite(val)):
                return None, np.inf, False
            norm = np.linalg.norm(val)
            return val, norm, True
        except Exception:
            return None, np.inf, False

    x0 = np.asarray(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    best_x = x0.copy()
    best_F = None
    best_norm = np.inf
    calls_made = 0

    # 1. Global Search Phase: Adaptive Radial Search
    
    # Check x0 first
    Fx, norm, ok = safe_eval_F(x0)
    calls_made += 1
    if ok:
        best_F = Fx
        best_norm = norm
        best_x = x0.copy()
        
    if best_norm <= 1e-8:
        return best_x

    # Adaptive search parameters
    # Start with a small radius, expand logarithmically.
    # Increase angular resolution as we get closer to the root (or as we expand).
    
    max_global_calls = 150  # Leave ~345 calls for Broyden
    
    # Log-spaced radii
    # We don't know the scale, so start wide and refine.
    # But since x0 is given, maybe the root is near x0?
    # Let's try radii: 1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5
    radii = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
    
    # For each radius, use a number of points.
    # Start with fewer points for large radii (coarse), more for small (fine).
    # Actually, usually we want to find the basin. 
    # Let's use a fixed set of angles but vary the count.
    
    # A better heuristic: 
    # If best_norm is decreasing, we are likely heading in the right direction.
    # But we are doing a blind search here.
    
    # Let's use a spiral-like search or just concentric rings.
    # To save calls, we can stop expanding if we find a very small norm.
    
    for r in radii:
        if not time_ok() or calls_made >= max_global_calls:
            break
            
        # Number of points: increase with smaller radius?
        # If r is small, we are close to x0.
        # If r is large, we are far.
        # Let's use 8 points for large, 16 for medium, 32 for small?
        # Or just a fixed 12-16 points per ring is usually enough for 2D.
        # Let's use 12 points.
        n_angles = 12
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        
        # Optimization: If r is very small, just check x0 + small perturbations?
        # No, the grid covers it.
        
        for theta in angles:
            if calls_made >= max_global_calls or not time_ok():
                break
            
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            x_cand = x0 + np.array([dx, dy])
            
            # Also check absolute points if x0 is near origin?
            # The current program does this. Let's keep it but make it conditional.
            # If x0 is close to 0, the ring around x0 covers the origin area.
            # If x0 is far, the ring around x0 might miss the origin.
            # Let's add a check for the origin if x0 is far away.
            # Actually, just checking x0 + offset is standard.
            # Let's stick to the ring around x0 for simplicity and consistency, 
            # unless x0 is exactly 0, in which case the ring covers the plane.
            
            Fx_c, norm_c, ok_c = safe_eval_F(x_cand)
            calls_made += 1
            
            if ok_c and norm_c < best_norm:
                best_norm = norm_c
                best_F = Fx_c
                best_x = x_cand.copy()
                
                if best_norm <= 1e-8:
                    return best_x

    # 2. Local Refinement Phase: Broyden's Method
    
    x = best_x.copy()
    Fx = best_F
    
    if Fx is None:
        Fx, norm, ok = safe_eval_F(x)
        calls_made += 1
        if not ok:
            return x

    # Broyden's Method
    h = 1e-6
    J = np.zeros((2, 2))
    
    # Initial Jacobian computation
    e0 = np.zeros(2)
    e0[0] = h
    x_p0 = x + e0
    Fx_p0, _, ok0 = safe_eval_F(x_p0)
    calls_made += 1
    if not ok0:
        return x
    J[:, 0] = (Fx_p0 - Fx) / h
    
    e1 = np.zeros(2)
    e1[1] = h
    x_p1 = x + e1
    Fx_p1, _, ok1 = safe_eval_F(x_p1)
    calls_made += 1
    if not ok1:
        return x
    J[:, 1] = (Fx_p1 - Fx) / h
    
    # Broyden iterations
    max_iters = 400
    
    for i in range(max_iters):
        if not time_ok() or calls_made >= 490:
            break
            
        if np.linalg.norm(Fx) <= 1e-12:
            return x
            
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            grad = J.T @ Fx
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-15:
                break
            dx = -grad / (norm_grad**2)
        
        x_new = x + dx
        Fx_new, norm_new, ok_new = safe_eval_F(x_new)
        calls_made += 1
        
        if not ok_new:
            x_half = x + 0.5 * dx
            Fx_half, norm_half, ok_half = safe_eval_F(x_half)
            calls_made += 1
            if ok_half and norm_half < np.linalg.norm(Fx):
                x_new = x_half
                Fx_new = Fx_half
            else:
                break
        
        if norm_new >= np.linalg.norm(Fx):
            improved = False
            for lam in [0.5, 0.25, 0.125]:
                x_trial = x + lam * dx
                Fx_trial, norm_trial, ok_trial = safe_eval_F(x_trial)
                calls_made += 1
                if ok_trial and norm_trial < np.linalg.norm(Fx):
                    x_new = x_trial
                    Fx_new = Fx_trial
                    improved = True
                    break
            if not improved:
                break
        
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        
        if s_norm_sq > 1e-12:
            Js = J @ s
            J = J + np.outer(y - Js, s) / s_norm_sq
        
        x = x_new
        Fx = Fx_new
        
        if np.linalg.norm(Fx) <= 1e-8:
            return x
            
    return x
