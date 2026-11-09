"""Root-finder for 3 fixed 2D systems using a multi-method hybrid approach.
Strategy: Broyden's Good Method with robust fallbacks, restart logic,
and a final polish phase using finite-difference Newton for high precision.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).
    
    Uses a hybrid approach:
    1. Broyden's Good Method (rank-1 update) as primary solver.
    2. Damped Newton with line search if Broyden stalls.
    3. Final polish phase with FD-Newton for high accuracy.
    4. Restart logic with perturbation if stuck.
    
    Budget: <500 F-calls per system. Time-limited to ~140s.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _time_ok():
        return time.time() < deadline

    def _call_F(x_val):
        nonlocal call_count
        if not _time_ok():
            return None
        if call_count >= 495:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _update_best(x_val, Fx_val):
        nonlocal best_x, best_norm
        norm = np.linalg.norm(Fx_val)
        if norm < best_norm:
            best_norm = norm
            best_x = x_val.copy()

    def _fd_jacobian(x_val, Fx_val, h=1e-5):
        """Compute Jacobian via forward differences. Uses 2 calls (Fx already known)."""
        J = np.zeros((2, 2))
        for j in range(2):
            if not _time_ok() or call_count >= 495:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x_val + e)
            if Fx_plus is not None:
                J[:, j] = (Fx_plus - Fx_val) / h
            else:
                J[:, j] = 1.0
        return J

    def _fd_jacobian_central(x_val, h=1e-5):
        """Compute Jacobian via central differences. Uses 4 calls."""
        J = np.zeros((2, 2))
        for j in range(2):
            if not _time_ok() or call_count >= 495:
                break
            e = np.zeros(2)
            e[j] = h
            Fp = _call_F(x_val + e)
            Fm = _call_F(x_val - e)
            if Fp is not None and Fm is not None:
                J[:, j] = (Fp - Fm) / (2 * h)
            else:
                J[:, j] = 1.0
        return J

    def _broyden_solve(J, Fx):
        """Solve J*dx = -Fx with regularization if needed."""
        try:
            return np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Try with regularization
            for lam in [1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
                try:
                    return np.linalg.solve(J + lam * np.eye(2), -Fx)
                except np.linalg.LinAlgError:
                    continue
            # Fallback to least squares
            try:
                return np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                return -Fx  # Steepest descent fallback

    def _broyden_iteration(x, Fx, J):
        """
        One iteration of Broyden's Good Method with line search.
        Returns (x_new, Fx_new, J_new, success)
        """
        nonlocal best_x, best_norm
        
        dx = _broyden_solve(J, Fx)
        
        if not np.all(np.isfinite(dx)):
            return x, Fx, J, False
        
        # Limit step size
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 50.0:
            dx = dx / norm_dx * 50.0
        
        norm_current = np.linalg.norm(Fx)
        
        # Backtracking line search
        alpha = 1.0
        for ls in range(25):
            if not _time_ok():
                break
                
            x_trial = x + alpha * dx
            if not np.all(np.isfinite(x_trial)):
                alpha *= 0.5
                continue
            
            Fx_trial = _call_F(x_trial)
            if Fx_trial is None:
                break
            
            norm_trial = np.linalg.norm(Fx_trial)
            _update_best(x_trial, Fx_trial)
            
            if norm_trial <= 1e-8:
                return x_trial, Fx_trial, J, True
            
            # Armijo condition
            if norm_trial < norm_current - 1e-4 * alpha * (dx @ Fx):
                # Broyden rank-1 update
                s = x_trial - x
                y = Fx_trial - Fx
                s_norm_sq = s.T @ s
                if s_norm_sq > 1e-15:
                    J = J + np.outer(y - J @ s, s) / s_norm_sq
                return x_trial, Fx_trial, J, True
            else:
                alpha *= 0.5
        
        return x, Fx, J, False

    def _damped_newton(x, Fx, max_iter=50):
        """
        Damped Newton method with line search.
        Recomputes Jacobian every few steps.
        """
        nonlocal best_x, best_norm, call_count
        
        J = _fd_jacobian(x, Fx)
        if not np.all(np.isfinite(J)):
            J = np.eye(2)
        
        for it in range(max_iter):
            if not _time_ok() or call_count >= 495:
                break
            
            dx = _broyden_solve(J, Fx)
            
            if not np.all(np.isfinite(dx)):
                break
            
            norm_dx = np.linalg.norm(dx)
            if norm_dx > 50.0:
                dx = dx / norm_dx * 50.0
            
            norm_current = np.linalg.norm(Fx)
            
            alpha = 1.0
            accepted = False
            for ls in range(20):
                if not _time_ok():
                    break
                    
                x_trial = x + alpha * dx
                if not np.all(np.isfinite(x_trial)):
                    alpha *= 0.5
                    continue
                
                Fx_trial = _call_F(x_trial)
                if Fx_trial is None:
                    break
                
                norm_trial = np.linalg.norm(Fx_trial)
                _update_best(x_trial, Fx_trial)
                
                if norm_trial <= 1e-8:
                    return x_trial, Fx_trial
                
                if norm_trial < norm_current:
                    x = x_trial
                    Fx = Fx_trial
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
            
            # Recompute Jacobian every 5 steps
            if (it + 1) % 5 == 0:
                J = _fd_jacobian(x, Fx)
                if not np.all(np.isfinite(J)):
                    J = np.eye(2)
        
        return x, Fx

    def _polish(x, Fx):
        """
        Final polish phase using central-difference Newton steps.
        """
        nonlocal best_x, best_norm, call_count
        
        for it in range(10):
            if not _time_ok() or call_count >= 495:
                break
            
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-8:
                return x, Fx
            
            J = _fd_jacobian_central(x)
            if not np.all(np.isfinite(J)):
                break
            
            dx = _broyden_solve(J, Fx)
            
            if not np.all(np.isfinite(dx)):
                break
            
            norm_dx = np.linalg.norm(dx)
            if norm_dx > 10.0:
                dx = dx / norm_dx * 10.0
            
            x_trial = x + dx
            if not np.all(np.isfinite(x_trial)):
                break
            
            Fx_trial = _call_F(x_trial)
            if Fx_trial is None:
                break
            
            norm_trial = np.linalg.norm(Fx_trial)
            _update_best(x_trial, Fx_trial)
            
            if norm_trial < norm_Fx:
                x = x_trial
                Fx = Fx_trial
            else:
                break
        
        return x, Fx

    # === Main Algorithm ===
    
    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        # Try a small perturbation
        for dx in [0.1, 0.01, 0.001]:
            x_pert = x + np.array([dx, dx])
            Fx = _call_F(x_pert)
            if Fx is not None:
                x = x_pert
                break
        if Fx is None:
            return best_x
    
    _update_best(x, Fx)
    if best_norm <= 1e-8:
        return best_x
    
    # Phase 1: Broyden's Good Method
    J = _fd_jacobian(x, Fx)
    if not np.all(np.isfinite(J)):
        J = np.eye(2)
    
    # Check conditioning
    try:
        cond_J = np.linalg.cond(J)
        if cond_J > 1e10:
            J = J + 1e-6 * np.eye(2)
    except np.linalg.LinAlgError:
        J = np.eye(2)
    
    stalled_count = 0
    prev_norm = best_norm
    
    for it in range(300):
        if not _time_ok() or call_count >= 495:
            break
        
        x, Fx, J, success = _broyden_iteration(x, Fx, J)
        
        if not success:
            stalled_count += 1
            if stalled_count >= 5:
                break
            continue
        
        stalled_count = 0
        norm_Fx = np.linalg.norm(Fx)
        
        if norm_Fx <= 1e-8:
            return x
        
        # Check for stagnation
        if norm_Fx > prev_norm * 0.99:
            stalled_count += 1
            if stalled_count >= 8:
                break
        
        prev_norm = norm_Fx
    
    # Phase 2: Damped Newton if Broyden didn't converge
    if best_norm > 1e-8:
        if not _time_ok() or call_count >= 495:
            return best_x
        
        x, Fx = _damped_newton(x, Fx, max_iter=50)
        _update_best(x, Fx)
    
    # Phase 3: Polish phase
    if best_norm > 1e-8:
        if not _time_ok() or call_count >= 495:
            return best_x
        
        x, Fx = _polish(x, Fx)
        _update_best(x, Fx)
    
    # Phase 4: If still stuck, try restart with perturbation
    if best_norm > 1e-8 and _time_ok() and call_count < 490:
        # Try a few random perturbations
        rng = np.random.default_rng(42)
        for attempt in range(3):
            if not _time_ok() or call_count >= 490:
                break
            
            scale = 0.1 * (10 ** (-attempt))
            perturbation = rng.uniform(-scale, scale, size=2)
            x_new = best_x + perturbation
            
            Fx_new = _call_F(x_new)
            if Fx_new is None:
                continue
            
            norm_new = np.linalg.norm(Fx_new)
            if norm_new < best_norm:
                x, Fx = x_new, Fx_new
                _update_best(x, Fx)
                
                # Quick Broyden from new point
                J = _fd_jacobian(x, Fx)
                if not np.all(np.isfinite(J)):
                    J = np.eye(2)
                
                for it in range(50):
                    if not _time_ok() or call_count >= 495:
                        break
                    
                    x, Fx, J, success = _broyden_iteration(x, Fx, J)
                    if not success:
                        break
                    if np.linalg.norm(Fx) <= 1e-8:
                        return x
                
                if best_norm <= 1e-8:
                    break
    
    return best_x
# EVOLVE-BLOCK-END
