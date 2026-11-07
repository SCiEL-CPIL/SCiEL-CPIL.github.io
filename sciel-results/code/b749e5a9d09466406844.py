"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    x = np.array(x0, dtype=float).reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() >= deadline
    
    # Initial evaluation
    Fx = np.asarray(F(x), dtype=float)
    if not np.all(np.isfinite(Fx)):
        return x
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    if best_norm <= 1e-8:
        return x
    
    # Finite-difference Jacobian (4 calls)
    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = np.asarray(F(x + e), dtype=float)
        if not np.all(np.isfinite(Fx_plus)):
            h = 1e-4
            Fx_plus = np.asarray(F(x + e), dtype=float)
            if not np.all(np.isfinite(Fx_plus)):
                # Fallback: just return best
                return best_x
        J[:, j] = (Fx_plus - Fx) / h
    
    # Broyden's method
    max_iters = 400  # We've used 5 calls, so 395 more max before 500
    for it in range(max_iters):
        if time_up():
            break
        
        # Check convergence
        if best_norm <= 1e-8:
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # J is singular, use pseudoinverse
            dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Try the full step
        x_trial = x + dx
        Fx_trial = np.asarray(F(x_trial), dtype=float)
        
        if not np.all(np.isfinite(Fx_trial)):
            # Step went to non-finite, backtrack
            alpha = 0.5
            success = False
            for _ in range(10):
                if time_up():
                    break
                x_trial = x + alpha * dx
                Fx_trial = np.asarray(F(x_trial), dtype=float)
                if np.all(np.isfinite(Fx_trial)):
                    trial_norm = np.linalg.norm(Fx_trial)
                    if trial_norm < best_norm:
                        x = x_trial
                        Fx = Fx_trial
                        best_x = x.copy()
                        best_norm = trial_norm
                        success = True
                        break
                alpha *= 0.5
            if not success:
                break
        else:
            trial_norm = np.linalg.norm(Fx_trial)
            if trial_norm < best_norm:
                # Good step, accept
                s = x_trial - x
                y = Fx_trial - Fx
                x = x_trial
                Fx = Fx_trial
                best_x = x.copy()
                best_norm = trial_norm
                
                # Broyden update
                if np.linalg.norm(s) > 1e-15:
                    yTs = np.dot(y, s)
                    if abs(yTs) > 1e-15:
                        J = J + np.outer(y - J @ s, s) / yTs
            else:
                # Step didn't improve, try line search
                alpha = 0.5
                accepted = False
                for _ in range(8):
                    if time_up():
                        break
                    x_trial = x + alpha * dx
                    Fx_trial = np.asarray(F(x_trial), dtype=float)
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    trial_norm = np.linalg.norm(Fx_trial)
                    if trial_norm < best_norm:
                        s = x_trial - x
                        y = Fx_trial - Fx
                        x = x_trial
                        Fx = Fx_trial
                        best_x = x.copy()
                        best_norm = trial_norm
                        
                        # Broyden update
                        if np.linalg.norm(s) > 1e-15:
                            yTs = np.dot(y, s)
                            if abs(yTs) > 1e-15:
                                J = J + np.outer(y - J @ s, s) / yTs
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    # No improvement possible with this direction, try perturbing J
                    # Reset to FD Jacobian
                    h2 = max(h, 1e-8)
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h2
                        Fx_plus = np.asarray(F(x + e), dtype=float)
                        if np.all(np.isfinite(Fx_plus)):
                            J[:, j] = (Fx_plus - Fx) / h2
                    h = h2
                    # Try again next iteration
                    continue
                if time_up():
                    break
        
        if best_norm <= 1e-8:
            break
    
    # Final check: return the best point found
    return best_x
# EVOLVE-BLOCK-END
