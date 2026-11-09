"""Root finder for 3 fixed two-dimensional systems using optimized Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with forward-difference Jacobian initialization.
    2. Strict call counting to maximize score (penalty for high mean calls).
    3. Adaptive Newton polishing for final precision.
    4. Stagnation detection to trigger restarts efficiently.
    5. Time-guarded to stay within 150s limit.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    x0 = np.array(x0, dtype=float)
    
    # Call tracking
    call_count = 0
    MAX_CALLS = 450  # Well under 500 to ensure success if converged
    
    def time_ok():
        return time.time() < deadline

    def F_counted(x):
        """Evaluate F with call counting and safety checks."""
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        call_count += 1
        try:
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return None
            return fx
        except:
            return None

    best_x = x0.copy()
    best_norm = np.inf
    
    def update_best(x, fx):
        """Update best solution if x is better."""
        nonlocal best_x, best_norm
        try:
            if fx is None:
                return
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    def get_jacobian_forward(x, Fx, h=None):
        """
        Estimate Jacobian using forward differences.
        Requires 2 additional F calls.
        """
        if h is None:
            h = 1e-7 * max(1.0, np.linalg.norm(x))
        
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        Fx1 = F_counted(x + e1)
        if Fx1 is None:
            return None
        Fx2 = F_counted(x + e2)
        if Fx2 is None:
            return None
            
        col1 = (Fx1 - Fx) / h
        col2 = (Fx2 - Fx) / h
        J = np.column_stack([col1, col2])
        
        if not np.all(np.isfinite(J)):
            return None
        return J

    def get_jacobian_central(x, h=None):
        """
        Estimate Jacobian using central differences.
        Requires 4 additional F calls.
        """
        if h is None:
            h = 1e-7 * max(1.0, np.linalg.norm(x))
        
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        Fx_p1 = F_counted(x + e1)
        Fx_m1 = F_counted(x - e1)
        Fx_p2 = F_counted(x + e2)
        Fx_m2 = F_counted(x - e2)
        
        if None in [Fx_p1, Fx_m1, Fx_p2, Fx_m2]:
            return None
            
        col1 = (Fx_p1 - Fx_m1) / (2 * h)
        col2 = (Fx_p2 - Fx_m2) / (2 * h)
        J = np.column_stack([col1, col2])
        
        if not np.all(np.isfinite(J)):
            return None
        return J

    def broyden(x_init, max_iter=40):
        """
        Broyden's Good Method.
        Returns (x, fx, success).
        """
        nonlocal call_count, best_norm, best_x
        
        x = np.array(x_init, dtype=float)
        Fx = F_counted(x)
        if Fx is None:
            return x, None, False
        
        n = np.linalg.norm(Fx)
        update_best(x, Fx)
        if n <= 1e-8:
            return x, Fx, True

        # Initialize Jacobian with forward differences
        J = get_jacobian_forward(x, Fx)
        if J is None:
            return x, Fx, False
            
        # Check initial singularity
        if abs(np.linalg.det(J)) < 1e-10:
            J += 1e-8 * np.eye(2)

        prev_norm = n
        stagnation_count = 0
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Solve for step
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, try pseudo-inverse
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            # Step damping: limit step size to prevent divergence
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
            elif step_norm < 1e-15:
                break
                
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                # Try smaller step
                dx = dx * 0.5
                x_new = x + dx
                if not np.all(np.isfinite(x_new)):
                    break

            Fx_new = F_counted(x_new)
            if Fx_new is None:
                break
                
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new <= 1e-8:
                return x_new, Fx_new, True
            
            # Stagnation check
            if n_new >= prev_norm * 0.95:
                stagnation_count += 1
                if stagnation_count >= 3:
                    break
            else:
                stagnation_count = 0
            prev_norm = n_new
            
            # Broyden Update
            s = x_new - x
            y = Fx_new - Fx
            denom = np.dot(s, s)
            if denom > 1e-30:
                J = J + np.outer(y - J @ s, s) / denom
                
            x = x_new
            Fx = Fx_new
            n = n_new

        return x, Fx, (n <= 1e-8)

    def polish(x_init, fx_init=None, max_iter=10):
        """
        Newton polishing for high precision.
        Uses central differences if close to root, else forward.
        """
        nonlocal call_count, best_norm, best_x
        
        x = np.array(x_init, dtype=float)
        
        if fx_init is None:
            Fx = F_counted(x)
            if Fx is None:
                return x, None, False
        else:
            Fx = np.asarray(fx_init, dtype=float)
            
        n = np.linalg.norm(Fx)
        if not np.isfinite(n):
            return x, Fx, False
            
        for _ in range(max_iter):
            if call_count >= MAX_CALLS - 10 or not time_ok():
                break
            if n <= 1e-8:
                return x, Fx, True
                
            # Choose difference scheme based on proximity
            if n < 1e-6:
                J = get_jacobian_central(x)
            else:
                J = get_jacobian_forward(x, Fx)
                
            if J is None:
                break
                
            if abs(np.linalg.det(J)) < 1e-10:
                J += 1e-8 * np.eye(2)
                
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
                
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break
                
            Fx_new = F_counted(x_new)
            if Fx_new is None:
                break
                
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new < n:
                x = x_new
                Fx = Fx_new
                n = n_new
            else:
                # Try half step
                x_half = x + 0.5 * dx
                Fx_half = F_counted(x_half)
                if Fx_half is not None:
                    n_half = np.linalg.norm(Fx_half)
                    if n_half < n:
                        x = x_half
                        Fx = Fx_half
                        n = n_half
                    else:
                        break
                else:
                    break
                    
        return x, Fx, (n <= 1e-8)

    # 1. Try Broyden from x0
    x_sol, fx_sol, success = broyden(x0)
    
    if success:
        # Polish to ensure we are well under 1e-8
        x_final, fx_final, _ = polish(x_sol, fx_sol)
        update_best(x_final, fx_final)
        return best_x

    # 2. If Broyden failed or stagnated, try restarts
    # Generate deterministic perturbations
    scale = max(1.0, np.linalg.norm(x0))
    perturbations = [
        np.array([1.0, 0.0]) * scale,
        np.array([0.0, 1.0]) * scale,
        np.array([1.0, 1.0]) * scale / np.sqrt(2),
        np.array([-1.0, 0.0]) * scale,
        np.array([0.0, -1.0]) * scale,
        np.array([1.0, -1.0]) * scale / np.sqrt(2),
        x0 * 0.5,
        x0 * 2.0,
        x0 * 0.1,
        x0 + np.array([scale, scale]),
        x0 + np.array([-scale, scale]),
        x0 + np.array([scale, -scale]),
        x0 + np.array([-scale, -scale]),
    ]
    
    for p in perturbations:
        if not time_ok() or call_count >= MAX_CALLS - 50:
            break
        x_init = x0 + p
        x_sol, fx_sol, success = broyden(x_init)
        
        if success:
            x_final, fx_final, _ = polish(x_sol, fx_sol)
            update_best(x_final, fx_final)
            return best_x

    # 3. If still not found, try a few random restarts if we have budget
    if call_count < MAX_CALLS - 50:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if not time_ok() or call_count >= MAX_CALLS - 50:
                break
            x_init = x0 + rng.uniform(-2*scale, 2*scale, size=2)
            x_sol, fx_sol, success = broyden(x_init)
            
            if success:
                x_final, fx_final, _ = polish(x_sol, fx_sol)
                update_best(x_final, fx_final)
                return best_x

    # 4. Final polish on the best candidate found so far
    if best_norm < np.inf and best_norm > 1e-8:
        if time_ok() and call_count < MAX_CALLS - 10:
            x_final, fx_final, _ = polish(best_x)
            update_best(x_final, fx_final)

    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
