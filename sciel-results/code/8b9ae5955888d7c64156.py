"""Root finder for 3 fixed 2-D systems: Global Grid Search + Broyden's Method."""
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Global Search: Scan a grid of points around x0 to find a starting point 
       with small ||F(x)||. This handles cases where x0 is far from the root.
    2. Local Refinement: Use Broyden's method (quasi-Newton) starting from the 
       best grid point. Broyden's method requires only 1 function evaluation per 
       iteration (after the initial Jacobian setup), allowing many more iterations 
       within the 500-call budget compared to finite-difference Newton.
    
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

    # 1. Global Search Phase
    # We try to find a starting point that is closer to the root.
    # We scan a set of radii and angles around x0.
    
    # Check x0 first
    Fx, norm, ok = safe_eval_F(x0)
    calls_made += 1
    if ok:
        best_F = Fx
        best_norm = norm
        best_x = x0.copy()
        
    if best_norm <= 1e-8:
        return best_x

    # Define search grid
    # Radii: logarithmic spacing from small to large
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    
    # Limit global search to ~100 calls max to leave room for Broyden
    max_global_calls = 100
    
    for r in radii:
        if not time_ok() or calls_made >= max_global_calls:
            break
        for theta in angles:
            if calls_made >= max_global_calls or not time_ok():
                break
            
            # Generate candidate point relative to x0
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            x_cand = x0 + np.array([dx, dy])
            
            # Also check absolute points if x0 is near origin to cover large basins
            if np.linalg.norm(x0) < 1.0:
                 x_cand2 = np.array([dx, dy])
            else:
                 x_cand2 = None

            # Evaluate x_cand
            Fx_c, norm_c, ok_c = safe_eval_F(x_cand)
            calls_made += 1
            
            if ok_c and norm_c < best_norm:
                best_norm = norm_c
                best_F = Fx_c
                best_x = x_cand.copy()
                
                if best_norm <= 1e-8:
                    return best_x
            
            # Evaluate x_cand2 if different
            if x_cand2 is not None and not np.allclose(x_cand, x_cand2):
                Fx_c2, norm_c2, ok_c2 = safe_eval_F(x_cand2)
                calls_made += 1
                if ok_c2 and norm_c2 < best_norm:
                    best_norm = norm_c2
                    best_F = Fx_c2
                    best_x = x_cand2.copy()
                    
                    if best_norm <= 1e-8:
                        return best_x
                        
            if not time_ok():
                break
        if not time_ok():
            break

    # 2. Local Refinement Phase: Broyden's Method
    # Start from best_x found in global search
    
    x = best_x.copy()
    Fx = best_F
    
    # If we didn't find a valid Fx, try x0 again or just return best_x
    if Fx is None:
        Fx, norm, ok = safe_eval_F(x)
        calls_made += 1
        if not ok:
            return x # Return whatever we have, likely failure but avoids exception

    # Broyden's Method
    # We need an initial Jacobian. We use finite differences for the first 3 calls.
    # Then we update J using the secant equation: J_new = J + (s_y - J s) s^T / (s^T s)
    # where s_y = F(x_new) - F(x_old) and s = x_new - x_old.
    
    h = 1e-6
    J = np.zeros((2, 2))
    
    # Initial Jacobian computation (3 calls: Fx is already known, need 2 more)
    # Column 0
    e0 = np.zeros(2)
    e0[0] = h
    x_p0 = x + e0
    Fx_p0, _, ok0 = safe_eval_F(x_p0)
    calls_made += 1
    if not ok0:
        return x
    J[:, 0] = (Fx_p0 - Fx) / h
    
    # Column 1
    e1 = np.zeros(2)
    e1[1] = h
    x_p1 = x + e1
    Fx_p1, _, ok1 = safe_eval_F(x_p1)
    calls_made += 1
    if not ok1:
        return x
    J[:, 1] = (Fx_p1 - Fx) / h
    
    # Broyden iterations
    max_iters = 450 # We have ~395 calls left (500 - 100 global - 3 init). 450 is safe.
    
    for i in range(max_iters):
        if not time_ok() or calls_made >= 490:
            break
            
        # Check convergence
        if np.linalg.norm(Fx) <= 1e-12:
            return x
            
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse or gradient step
            # Gradient step on ||F||^2: grad = J^T F
            grad = J.T @ Fx
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-15:
                break
            dx = -grad / (norm_grad**2) # Newton step for gradient
        
        # Damping/Line Search: We want to ensure we don't overshoot or diverge.
        # Broyden can be unstable if the step is too large.
        # We try full step first. If it fails to reduce norm, we dampen.
        
        x_new = x + dx
        Fx_new, norm_new, ok_new = safe_eval_F(x_new)
        calls_made += 1
        
        if not ok_new:
            # If evaluation failed, try a smaller step
            # This is a fallback, Broyden usually works with full step if J is good
            # Try half step
            x_half = x + 0.5 * dx
            Fx_half, norm_half, ok_half = safe_eval_F(x_half)
            calls_made += 1
            if ok_half and norm_half < np.linalg.norm(Fx):
                x_new = x_half
                Fx_new = Fx_half
            else:
                # If even half step fails or doesn't improve, break or return best
                break
        
        # Check if we made progress
        if norm_new >= np.linalg.norm(Fx):
            # No improvement. Try damping.
            # Simple bisection for lambda
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
        
        # Update Jacobian using Broyden's formula
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        
        if s_norm_sq > 1e-12:
            # J = J + (y - J s) s^T / (s^T s)
            Js = J @ s
            J = J + np.outer(y - Js, s) / s_norm_sq
        
        x = x_new
        Fx = Fx_new
        
        if np.linalg.norm(Fx) <= 1e-8:
            return x
            
    return x
