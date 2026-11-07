"""Root finder for 3 fixed 2D systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method (good for small n) with line search safeguards
    and a deadline guard to avoid timeouts.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    x = np.array(x0, dtype=float)
    if x.shape != (2,):
        x = x.reshape(2)

    # Initial function evaluation
    try:
        Fx = F(x)
    except:
        # Try a small perturbation
        x = x + np.array([1e-3, -1e-3])
        try:
            Fx = F(x)
        except:
            return x

    if not np.all(np.isfinite(Fx)):
        # Try a small perturbation
        x = x + np.array([1e-3, -1e-3])
        try:
            Fx = F(x)
        except:
            return x
        if not np.all(np.isfinite(Fx)):
            return x

    # Check if already at root
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Initialize Jacobian using finite differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-6 * max(1.0, abs(x[j]))
            try:
                Fx_pert = F(x + e)
            except:
                return J
            if not np.all(np.isfinite(Fx_pert)):
                return J
            J[:, j] = (Fx_pert - Fx) / e[j]
        return J

    J = compute_jacobian(x, Fx)

    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    # Main Broyden loop
    for iteration in range(100):
        if time.time() > deadline:
            break

        # Check convergence
        if best_norm <= 1e-8:
            return best_x

        # Try to solve J dx = -Fx
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Reset Jacobian
            J = compute_jacobian(x, Fx)
            try:
                dx = -np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                break

        if not np.all(np.isfinite(dx)):
            break

        # Line search with backtracking
        alpha = 1.0
        Fx_old = Fx.copy()
        norm_old = np.linalg.norm(Fx)

        for ls_iter in range(20):
            x_trial = x + alpha * dx
            try:
                Fx_trial = F(x_trial)
            except:
                alpha *= 0.5
                continue
            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue
            norm_trial = np.linalg.norm(Fx_trial)

            # Accept if we improved
            if norm_trial < norm_old:
                x = x_trial
                Fx = Fx_trial
                best_x = x.copy()
                best_norm = norm_trial
                if best_norm <= 1e-8:
                    return x

                # Update Jacobian (Broyden update)
                s = x - (x - alpha * dx)  # = alpha * dx
                y = Fx - Fx_old
                if np.linalg.norm(s) > 1e-15:
                    J = J + np.outer(y - J @ s, s) / (s @ s)
                break
            else:
                alpha *= 0.5

        else:
            # No improvement in line search, try a different direction
            # Use steepest descent direction
            grad = J.T @ Fx
            if np.linalg.norm(grad) > 1e-15:
                dx = -grad / np.linalg.norm(grad)
                alpha = 1.0
                for ls_iter in range(10):
                    x_trial = x + alpha * dx
                    try:
                        Fx_trial = F(x_trial)
                    except:
                        alpha *= 0.5
                        continue
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    norm_trial = np.linalg.norm(Fx_trial)
                    if norm_trial < norm_old:
                        x = x_trial
                        Fx = Fx_trial
                        best_x = x.copy()
                        best_norm = norm_trial
                        if best_norm <= 1e-8:
                            return x
                        # Reset Jacobian
                        J = compute_jacobian(x, Fx)
                        break
                    alpha *= 0.5
                if norm_trial >= norm_old:
                    break
            else:
                break

    # If we didn't converge, try a few more Newton steps with fresh Jacobians
    for _ in range(20):
        if time.time() > deadline:
            break
        try:
            Fx = F(x)
        except:
            break
        if not np.all(np.isfinite(Fx)):
            break
        norm = np.linalg.norm(Fx)
        if norm <= 1e-8:
            return x
        if norm < best_norm:
            best_x = x.copy()
            best_norm = norm
        J = compute_jacobian(x, Fx)
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dx)):
            break
        x = x + dx

    # If still not converged, try restarts with diverse starting points
    if time.time() < deadline - 5.0 and best_norm > 1e-8:
        # Generate diverse starting points
        starts = [x0.copy(), best_x.copy()]
        
        # Add perturbations around best_x
        for scale in [0.1, 0.5, 1.0, 5.0, 10.0]:
            for dx_val in [0.1, -0.1, 0.5, -0.5]:
                s = best_x.copy()
                s[0] += dx_val * scale
                s[1] += dx_val * scale
                starts.append(s)
        
        # Add some random-looking starts based on x0
        for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 7*np.pi/4]:
            r = np.linalg.norm(x0)
            if r < 1e-10:
                r = 1.0
            s = np.array([r * np.cos(angle), r * np.sin(angle)])
            starts.append(s)
        
        for s in starts:
            if time.time() > deadline - 2.0:
                break
            
            if not np.all(np.isfinite(s)):
                continue
            
            # Try Broyden from this start
            x_s = s.copy()
            try:
                Fx_s = F(x_s)
            except:
                continue
            if not np.all(np.isfinite(Fx_s)):
                continue
            norm_s = np.linalg.norm(Fx_s)
            if norm_s <= 1e-8:
                return x_s
            if norm_s < best_norm:
                best_x = x_s.copy()
                best_norm = norm_s
            
            J_s = compute_jacobian(x_s, Fx_s)
            for _ in range(30):
                if time.time() > deadline - 1.0:
                    break
                try:
                    dx_s = -np.linalg.solve(J_s, Fx_s)
                except np.linalg.LinAlgError:
                    J_s = compute_jacobian(x_s, Fx_s)
                    try:
                        dx_s = -np.linalg.solve(J_s, Fx_s)
                    except np.linalg.LinAlgError:
                        break
                if not np.all(np.isfinite(dx_s)):
                    break
                
                alpha_s = 1.0
                for _ in range(15):
                    x_trial_s = x_s + alpha_s * dx_s
                    try:
                        Fx_trial_s = F(x_trial_s)
                    except:
                        alpha_s *= 0.5
                        continue
                    if not np.all(np.isfinite(Fx_trial_s)):
                        alpha_s *= 0.5
                        continue
                    norm_trial_s = np.linalg.norm(Fx_trial_s)
                    if norm_trial_s < norm_s:
                        s_old = x_s.copy()
                        Fx_old_s = Fx_s.copy()
                        x_s = x_trial_s
                        Fx_s = Fx_trial_s
                        norm_s = norm_trial_s
                        if norm_s <= 1e-8:
                            return x_s
                        if norm_s < best_norm:
                            best_x = x_s.copy()
                            best_norm = norm_s
                        # Broyden update
                        sv = x_s - s_old
                        yv = Fx_s - Fx_old_s
                        if np.linalg.norm(sv) > 1e-15:
                            J_s = J_s + np.outer(yv - J_s @ sv, sv) / (sv @ sv)
                        break
                    alpha_s *= 0.5

    # Return the best result we found
    if np.all(np.isfinite(best_x)):
        return best_x
    
    return x0.copy()
# EVOLVE-BLOCK-END
