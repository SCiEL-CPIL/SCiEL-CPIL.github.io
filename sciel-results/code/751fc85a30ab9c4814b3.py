"""Root finder for 3 fixed 2-D systems: Secant-Step Hybrid with Broyden re-anchoring."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Secant-Step Hybrid:
       - Initialize Jacobian via a two-point secant difference (costing 2 calls).
       - Alternate between standard Broyden rank-1 updates and a full 3-call 
         finite-difference Jacobian recomputation every k iterations to correct 
         accumulated approximation error. This "re-anchoring" strategy prevents 
         the quasi-Newton matrix from drifting into ill-conditioning.
    2. Line search for robustness.
    3. Final polish with Newton's method when close to root.
    4. Adaptive restarts from promising points if the first attempt fails.
    
    Budget:
    - Time limit: ~115s internal (evaluator kills at 150s).
    - Call limit: < 500 calls per system.
    """
    start_time = time.time()
    time_limit = 115.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    call_count = [0]
    
    def safe_eval_F(x):
        """Evaluate F with error handling."""
        try:
            x_arr = np.array(x, dtype=float)
            if x_arr.shape != (2,) or not np.all(np.isfinite(x_arr)):
                call_count[0] += 1
                return None
            result = np.array(F(x_arr), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None
    
    def compute_jacobian_fd(x, Fx, h=1e-7):
        """Compute Jacobian using forward finite differences. Costs 2 calls."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def compute_jacobian_central(x, h=1e-6):
        """Compute Jacobian using central finite differences. Costs 4 calls."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            Fx_minus = safe_eval_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                return None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J
    
    def broyden_secant_hybrid(x_start, max_iter=40, reanchor_every=5, call_budget=450):
        """
        Secant-Step Hybrid Broyden's method.
        Returns (x, Fx, success).
        """
        x = np.array(x_start, dtype=float).copy()
        
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian: Secant step
        # We need a second point to form a secant. 
        # We perturb x in the direction of -Fx (steepest descent of ||F||^2/2 approx) 
        # or just a small random/axis perturbation. 
        # Using a small step in the direction of the first column of identity 
        # or based on Fx is common. Let's use a small step in the direction 
        # that is orthogonal to Fx if possible, or just axis-aligned.
        # To be robust, let's try a small step in the direction of the 
        # normalized Fx (if non-zero) or e1.
        
        Fx_norm = np.linalg.norm(Fx)
        if Fx_norm > 1e-10:
            # Step in direction of -Fx (like a gradient step for ||F||)
            # But for Jacobian estimation, we just need a delta x.
            # Let's use a small step in the direction of the first basis vector 
            # scaled by the norm of x or a small constant.
            # Actually, a better secant initialization for 2D:
            # Pick a direction d. Let's use d = -Fx / ||Fx|| * small_step
            # Or simply d = h * e1.
            # The prompt suggests "two-point secant difference".
            # Let's try a step in the direction of the normalized Fx.
            d = -Fx / Fx_norm
        else:
            d = np.array([1.0, 0.0])
        
        # Choose a small step size for the secant
        # If x is large, relative step. If x is small, absolute.
        x_norm = np.linalg.norm(x)
        step_size = max(1e-5, 1e-5 * x_norm)
        
        x_secant = x + step_size * d
        Fx_secant = safe_eval_F(x_secant)
        
        if Fx_secant is None:
            # Fallback to FD if secant point is invalid
            J = compute_jacobian_fd(x, Fx, h=1e-7)
            if J is None:
                return x, Fx, False
        else:
            # Secant Jacobian: J * (x2 - x1) = F2 - F1
            # J = (F2 - F1) * (x2 - x1)^T / ||x2 - x1||^2  (Rank 1 update from identity? No.)
            # For 2D, we have 2 equations and 4 unknowns. 
            # Standard approach: Start with an identity or FD Jacobian?
            # The prompt says "initializes the Jacobian via a two-point secant difference".
            # This usually implies using the secant to update an initial guess.
            # If we assume J0 = I, then J1 = I + (F2-F1 - I*(x2-x1)) * (x2-x1)^T / ||x2-x1||^2
            # This is a Broyden update from J0=I.
            # Let's do that.
            J = np.eye(2)
            s = x_secant - x
            y = Fx_secant - Fx
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s
            
            # Check if J is invertible. If not, fallback to FD.
            try:
                np.linalg.inv(J)
            except np.linalg.LinAlgError:
                J = compute_jacobian_fd(x, Fx, h=1e-7)
                if J is None:
                    return x, Fx, False

        if abs(np.linalg.det(J)) < 1e-12:
            # Fallback to FD if singular
            J_fd = compute_jacobian_fd(x, Fx, h=1e-7)
            if J_fd is not None and abs(np.linalg.det(J_fd)) > 1e-12:
                J = J_fd
            else:
                return x, Fx, False

        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        jacobian_age = 0
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_budget:
                break
            
            # Re-anchoring strategy
            jacobian_age += 1
            should_reanchor = False
            if jacobian_age >= reanchor_every:
                should_reanchor = True
            
            if should_reanchor:
                # Recompute Jacobian via FD (3 calls? No, FD is 2 calls for forward, 4 for central).
                # The prompt says "full 3-call finite-difference Jacobian recomputation".
                # Standard FD for 2x2 is 2 calls (forward) or 4 (central).
                # Maybe they mean 1 for F(x) + 2 for derivatives? But we already have F(x).
                # So 2 calls for derivatives. Total 3 if counting the current F(x) which we have.
                # Let's use forward FD (2 calls).
                J_new = compute_jacobian_fd(x, Fx, h=1e-7)
                if J_new is not None and abs(np.linalg.det(J_new)) > 1e-12:
                    J = J_new
                    jacobian_age = 0
                else:
                    # Try central if forward fails or is singular
                    J_new = compute_jacobian_central(x, h=1e-6)
                    if J_new is not None and abs(np.linalg.det(J_new)) > 1e-12:
                        J = J_new
                        jacobian_age = 0
            
            # Solve for step
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, try re-anchoring immediately
                J_new = compute_jacobian_fd(x, Fx, h=1e-5)
                if J_new is not None:
                    J = J_new
                    try:
                        dx = np.linalg.solve(J, -Fx)
                    except np.linalg.LinAlgError:
                        break
                else:
                    break
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)
            
            # Line search
            alpha = 1.0
            found_step = False
            for _ in range(8):
                x_new = x + alpha * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    alpha *= 0.5
                    continue
                
                Fx_new_norm = np.linalg.norm(Fx_new)
                
                # Accept if improvement
                if Fx_new_norm < best_norm * 0.9 or Fx_new_norm < best_norm:
                    found_step = True
                    break
                
                alpha *= 0.5
            
            if not found_step:
                # If no step found, break
                break
            
            x_new = x + alpha * dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm
            
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            # Update Jacobian using Broyden's formula (if not re-anchored this step)
            if not should_reanchor:
                s = x_new - x
                y = Fx_new - Fx
                s_dot_s = np.dot(s, s)
                if s_dot_s > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
            
            # Polish check
            if best_norm < 1e-4 and call_count[0] < call_budget - 20:
                # Try a few Newton steps with central diff
                for _ in range(5):
                    if not time_ok() or call_count[0] >= call_budget:
                        break
                    J_c = compute_jacobian_central(x, h=1e-6)
                    if J_c is None:
                        break
                    try:
                        dx_n = np.linalg.solve(J_c, -Fx)
                    except np.linalg.LinAlgError:
                        break
                    x_n = x + dx_n
                    Fx_n = safe_eval_F(x_n)
                    if Fx_n is None:
                        break
                    norm_n = np.linalg.norm(Fx_n)
                    if norm_n <= 1e-8:
                        return x_n, Fx_n, True
                    if norm_n < best_norm:
                        best_x = x_n.copy()
                        best_Fx = Fx_n.copy()
                        best_norm = norm_n
                        x = x_n
                        Fx = Fx_n
                    else:
                        break
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    # First attempt: Secant-Step Hybrid from x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden_secant_hybrid(x0_arr, max_iter=40, reanchor_every=5, call_budget=450)
    if success:
        return x
    
    # Adaptive restarts
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    # Generate diverse restart points
    restarts = []
    n0_norm = np.linalg.norm(x0_arr)
    
    if n0_norm < 1e-10:
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            restarts.append(np.array([np.cos(angle), np.sin(angle)]))
    else:
        for scale in [0.5, 2.0, 0.1, 5.0, -1.0, -0.5, 0.2, 10.0]:
            restarts.append(scale * x0_arr)
        
        perp1 = np.array([-x0_arr[1], x0_arr[0]])
        perp2 = np.array([x0_arr[1], -x0_arr[0]])
        for p in [perp1, perp2]:
            p_norm = np.linalg.norm(p)
            if p_norm > 1e-10:
                p = p / p_norm * n0_norm
                restarts.append(p)
                restarts.append(x0_arr + p)
                restarts.append(x0_arr - p)
    
    Fx0 = safe_eval_F(x0_arr)
    if Fx0 is not None:
        Fx0_norm = np.linalg.norm(Fx0)
        if Fx0_norm > 1e-10:
            direction = -Fx0 / Fx0_norm
            for step in [0.1, 0.5, 1.0, 2.0]:
                restarts.append(x0_arr + step * n0_norm * direction)
    
    rng = np.random.RandomState(42)
    for _ in range(8):
        restarts.append(rng.randn(2) * max(1.0, n0_norm))
    
    restarts.append(np.zeros(2))
    
    for r in restarts:
        if not time_ok() or call_count[0] >= 470:
            break
        
        x_r, Fx_r, success_r = broyden_secant_hybrid(r, max_iter=30, reanchor_every=5, call_budget=470)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
    
    return best_x
