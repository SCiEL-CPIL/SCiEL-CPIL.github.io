"""Root finder for 3 fixed 2-D systems: Secant-Step Hybrid Broyden with re-anchoring and polish."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Secant-Step Hybrid Broyden: 2-call secant Jacobian init, rank-1 updates,
       periodic 3-call FD re-anchoring to prevent drift.
    2. Damped line search for robustness.
    3. Multiple diverse restarts if initial attempt fails.
    4. Final polish step with high-precision FD Jacobian if close.
    """
    start_time = time.time()
    time_limit = 118.0  # Safety margin before 150s kill
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    # Safe F evaluation with call counting
    call_count = [0]
    call_limit = 490  # Stay under 500
    
    def safe_eval_F(x):
        """Evaluate F with error handling and call counting."""
        try:
            result = np.array(F(x), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None
    
    def compute_J_secant(x, Fx):
        """
        Compute Jacobian via 2-point secant (2 calls).
        Uses a perturbation in the direction of Fx (or x0 direction if Fx is small).
        """
        # Choose perturbation direction
        fx_norm = np.linalg.norm(Fx)
        if fx_norm > 1e-10:
            direction = -Fx / fx_norm
        else:
            direction = np.array([1.0, 0.0])
        
        # Choose step size
        h = 1e-5
        
        x_pert = x + h * direction
        Fx_pert = safe_eval_F(x_pert)
        if Fx_pert is None:
            return None
        
        # Secant estimate: J * (h*direction) ≈ Fx_pert - Fx
        # J = (Fx_pert - Fx) / h * inv(direction)
        # For 2D, we need another direction
        # Use perpendicular direction
        perp = np.array([-direction[1], direction[0]])
        x_pert2 = x + h * perp
        Fx_pert2 = safe_eval_F(x_pert2)
        if Fx_pert2 is None:
            return None
        
        # J * [h*dir, h*perp] = [Fx_pert - Fx, Fx_pert2 - Fx]
        delta_x = h * np.column_stack([direction, perp])
        delta_f = np.column_stack([Fx_pert - Fx, Fx_pert2 - Fx])
        
        try:
            J = delta_f @ np.linalg.inv(delta_x)
        except np.linalg.LinAlgError:
            return None
        
        if not np.all(np.isfinite(J)):
            return None
            
        return J
    
    def compute_J_fd(x, Fx, h=1e-6):
        """Compute Jacobian via forward finite differences (2 calls)."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def compute_J_central(x, h=1e-6):
        """Compute Jacobian via central finite differences (4 calls)."""
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
    
    def broyden_hybrid(x_start, max_iter=35, reanchor_every=5):
        """
        Secant-Step Hybrid Broyden with periodic re-anchoring.
        Returns (x, Fx, success).
        """
        x = np.array(x_start, dtype=float)
        
        # Initial evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian via secant (2 calls)
        J = compute_J_secant(x, Fx)
        if J is None:
            # Fallback to FD
            J = compute_J_fd(x, Fx, 1e-5)
            if J is None:
                return x, Fx, False
        
        # Check invertibility
        try:
            np.linalg.solve(J, np.ones(2))
        except np.linalg.LinAlgError:
            # Try different perturbation
            J = compute_J_fd(x, Fx, 1e-4)
            if J is None:
                return x, Fx, False
            try:
                np.linalg.solve(J, np.ones(2))
            except np.linalg.LinAlgError:
                return x, Fx, False
        
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_limit:
                break
            
            # Periodic re-anchoring: recompute full Jacobian
            if it > 0 and it % reanchor_every == 0:
                J_new = compute_J_fd(x, Fx, 1e-6)
                if J_new is not None:
                    J = J_new
                else:
                    # Try central differences
                    J_new = compute_J_central(x, 1e-6)
                    if J_new is not None:
                        J = J_new
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Recompute Jacobian
                J = compute_J_fd(x, Fx, 1e-5)
                if J is None:
                    break
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 30.0:
                dx = dx * (30.0 / step_norm)
            
            # Line search
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            
            if Fx_new is None:
                for alpha in [0.5, 0.25, 0.1, 0.05]:
                    x_new = x + alpha * dx
                    Fx_new = safe_eval_F(x_new)
                    if Fx_new is not None:
                        break
                if Fx_new is None:
                    break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            # Track best
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm
            
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            # Broyden rank-1 update
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    def polish(x_start, Fx_start):
        """
        Final polish using Newton's method with high-precision FD Jacobian.
        Only called if we're close to a root.
        """
        x = np.array(x_start, dtype=float)
        Fx = np.array(Fx_start, dtype=float)
        
        fx_norm = np.linalg.norm(Fx)
        if fx_norm > 1e-4:
            return x, Fx, False
        
        for it in range(10):
            if not time_ok() or call_count[0] >= call_limit:
                break
            
            # Central difference Jacobian for accuracy
            J = compute_J_central(x, 1e-7)
            if J is None:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            fx_new_norm = np.linalg.norm(Fx_new)
            if fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            if fx_new_norm < np.linalg.norm(Fx):
                x = x_new
                Fx = Fx_new
            else:
                # Step too large, damp
                for alpha in [0.5, 0.25, 0.1]:
                    x_try = x + alpha * dx
                    Fx_try = safe_eval_F(x_try)
                    if Fx_try is not None and np.linalg.norm(Fx_try) < np.linalg.norm(Fx):
                        x = x_try
                        Fx = Fx_try
                        break
                else:
                    break
        
        return x, Fx, np.linalg.norm(Fx) <= 1e-8
    
    # Main search
    x0_arr = np.array(x0, dtype=float)
    
    # First attempt from x0
    x, Fx, success = broyden_hybrid(x0_arr, max_iter=35, reanchor_every=5)
    if success:
        return x
    
    # If close, try polish
    if Fx is not None and np.linalg.norm(Fx) < 1e-4:
        x_pol, Fx_pol, success_pol = polish(x, Fx)
        if success_pol:
            return x_pol
        if Fx_pol is not None and np.linalg.norm(Fx_pol) < (np.linalg.norm(Fx) if Fx is not None else np.inf):
            x, Fx = x_pol, Fx_pol
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    # Generate diverse restarts
    restarts = []
    x0_norm = np.linalg.norm(x0_arr)
    scale = max(x0_norm, 1.0)
    
    # Scaled versions
    for s in [0.5, 2.0, 0.1, 5.0, -1.0, -0.5, 0.01, 10.0]:
        restarts.append(s * x0_arr)
    
    # Origin
    restarts.append(np.zeros(2))
    
    # Random perturbations
    rng = np.random.RandomState(12345)
    for _ in range(12):
        restarts.append(x0_arr + rng.randn(2) * scale * 0.7)
    
    # Pure random in reasonable range
    for _ in range(6):
        restarts.append(rng.randn(2) * scale)
    
    # Try each restart
    for r in restarts:
        if not time_ok() or call_count[0] >= call_limit - 30:
            break
        
        x_r, Fx_r, success_r = broyden_hybrid(r, max_iter=25, reanchor_every=5)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
                
                # Try polish if close
                if norm_r < 1e-4:
                    x_pol, Fx_pol, success_pol = polish(x_r, Fx_r)
                    if success_pol:
                        return x_pol
                    if Fx_pol is not None and np.linalg.norm(Fx_pol) < best_norm:
                        best_norm = np.linalg.norm(Fx_pol)
                        best_x = x_pol.copy()
    
    # Final: try polish on best
    if Fx is not None and best_norm < 1e-4:
        x_pol, Fx_pol, success_pol = polish(best_x, Fx)
        if success_pol:
            return x_pol
        if Fx_pol is not None and np.linalg.norm(Fx_pol) < best_norm:
            best_x = x_pol.copy()
    
    return best_x
