"""Root finder for 3 fixed 2D systems using homotopy/continuation with Newton refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses homotopy/continuation from a known anchor, then Newton refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    x = np.array(x0, dtype=float)
    if x.shape != (2,):
        x = x.reshape(2)

    # Initial function evaluation
    Fx = np.array(F(x), dtype=float)
    n_calls = 1
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    if best_norm <= 1e-8:
        return x

    # Helper: compute FD Jacobian
    def compute_fd_jacobian(x_curr, F_curr):
        J_fd = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = np.array(F(x_curr + e), dtype=float)
            J_fd[:, j] = (Fx_e - F_curr) / h
        return J_fd

    # Helper: Newton step with line search
    def newton_step(x_curr, F_curr, J_curr, max_ls=10):
        nonlocal n_calls
        try:
            s = -np.linalg.solve(J_curr, F_curr)
        except np.linalg.LinAlgError:
            J_curr = compute_fd_jacobian(x_curr, F_curr)
            n_calls += 2
            try:
                s = -np.linalg.solve(J_curr, F_curr)
            except np.linalg.LinAlgError:
                return None, None, J_curr

        step_norm = np.linalg.norm(s)
        if step_norm > 10.0:
            s = s / step_norm * 10.0

        for ls in range(max_ls):
            alpha = 1.0 / (2 ** ls)
            x_trial = x_curr + alpha * s
            F_trial = np.array(F(x_trial), dtype=float)
            n_calls += 1
            if not np.all(np.isfinite(F_trial)):
                continue
            norm_trial = np.linalg.norm(F_trial)
            if norm_trial < np.linalg.norm(F_curr):
                # Broyden update
                s_act = x_trial - x_curr
                y = F_trial - F_curr
                s_dot_s = np.dot(s_act, s_act)
                if s_dot_s > 1e-15:
                    r = y - J_curr @ s_act
                    J_curr = J_curr + np.outer(r, s_act) / s_dot_s
                return x_trial, F_trial, J_curr
        return None, None, J_curr

    # Homotopy/continuation approach
    # We blend between a simple solvable system and the target F
    # Anchor: F_anchor(x) = x (root at origin)
    # Blend: G_t(x) = (1-t) * x + t * F(x)
    # Start at t=0 where root is x=0, incrementally increase t

    # First, try direct Newton from x0
    J = compute_fd_jacobian(x, Fx)
    n_calls += 2

    converged = False
    for it in range(30):
        if time.time() > deadline:
            break
        if best_norm <= 1e-8:
            converged = True
            break
        result = newton_step(x, Fx, J)
        if result[0] is not None:
            x, Fx, J = result
            norm_now = np.linalg.norm(Fx)
            if norm_now < best_norm:
                best_x = x.copy()
                best_norm = norm_now
            if best_norm <= 1e-8:
                converged = True
                break
        else:
            # Recompute Jacobian and try again
            J = compute_fd_jacobian(x, Fx)
            n_calls += 2

    if converged:
        return best_x

    # If direct Newton failed, try homotopy/continuation
    # Blend: G_t(x) = (1-t)*x + t*F(x)
    # At t=0, root is x=0. We trace the path as t increases.
    
    # Start from origin (root of anchor system)
    x_h = np.array([0.0, 0.0])
    
    # Try to find where F(x0) is small enough to use x0 as anchor
    # Otherwise use origin
    
    # Incremental t values
    t_values = np.linspace(0.0, 1.0, 15)
    
    for t in t_values:
        if time.time() > deadline - 10.0:
            break
        
        # Define blended function
        def G_t(x_curr, t_val=t):
            F_val = np.array(F(x_curr), dtype=float)
            return (1.0 - t_val) * x_curr + t_val * F_val
        
        # Evaluate at current x_h
        Gx = G_t(x_h)
        n_calls += 1  # Count the F call inside G_t
        
        if not np.all(np.isfinite(Gx)):
            continue
        
        norm_G = np.linalg.norm(Gx)
        if norm_G <= 1e-8:
            # This is a root of G_t, which is close to a root of F when t is large
            # Check if it's also a root of F
            Fx_check = np.array(F(x_h), dtype=float)
            n_calls += 1
            if np.all(np.isfinite(Fx_check)):
                norm_F = np.linalg.norm(Fx_check)
                if norm_F < best_norm:
                    best_x = x_h.copy()
                    best_norm = norm_F
                if norm_F <= 1e-8:
                    return x_h
        
        # Newton step on G_t
        # Compute Jacobian of G_t: (1-t)*I + t*J_F
        J_F = compute_fd_jacobian(x_h, np.array(F(x_h), dtype=float))
        n_calls += 2
        J_G = (1.0 - t) * np.eye(2) + t * J_F
        
        try:
            s = -np.linalg.solve(J_G, Gx)
        except np.linalg.LinAlgError:
            continue
        
        # Line search on G_t
        accepted = False
        for ls in range(8):
            alpha = 1.0 / (2 ** ls)
            x_trial = x_h + alpha * s
            G_trial = G_t(x_trial)
            n_calls += 1  # F call inside G_t
            
            if not np.all(np.isfinite(G_trial)):
                continue
            
            norm_trial = np.linalg.norm(G_trial)
            if norm_trial < norm_G:
                x_h = x_trial
                Gx = G_trial
                accepted = True
                break
        
        if not accepted:
            # Try smaller step
            x_trial = x_h - 0.1 * Gx
            G_trial = G_t(x_trial)
            n_calls += 1
            if np.all(np.isfinite(G_trial)) and np.linalg.norm(G_trial) < norm_G:
                x_h = x_trial
                Gx = G_trial

    # Final check
    if best_norm <= 1e-8:
        return best_x

    # Fallback: SciPy root
    if time.time() < deadline - 5.0:
        try:
            from scipy.optimize import root as scipy_root

            def F_wrapped(x):
                nonlocal n_calls
                if time.time() > deadline - 2.0:
                    return np.array([0.0, 0.0])
                result = np.array(F(x), dtype=float)
                n_calls += 1
                return result

            result = scipy_root(F_wrapped, best_x, method='hybr', tol=1e-10, maxfev=30)
            if result.success and np.all(np.isfinite(result.x)):
                Fx_result = np.array(F(result.x), dtype=float)
                n_calls += 1
                result_norm = np.linalg.norm(Fx_result)
                if result_norm < best_norm:
                    best_x = result.x.copy()
                    best_norm = result_norm
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
