"""Root-finder for 3 fixed 2D systems using Broyden's method with a robust SciPy fallback.
Strategy:
  1. Broyden's method (good variant) with damped steps for fast local convergence.
  2. If Broyden stagnates or fails, switch to scipy.optimize.minimize with method='dogleg'
     on g(x) = 0.5*||F(x)||^2, providing a finite-difference-based gradient.
  3. Final polish with FD-Newton steps for high precision.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).
    
    Uses Broyden's method with:
    - Forward-difference initial Jacobian
    - Damped Newton steps with adaptive step control
    - Robust fallback to scipy.optimize.minimize (dogleg) if Broyden stalls
    - Final polish phase with FD-Newton for high accuracy
    
    Budget: <500 F-calls per system. Time-limited to ~135s.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 135.0
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _time_ok():
        return time.time() < deadline

    def _call_F(x_val):
        nonlocal call_count
        if not _time_ok() or call_count >= 495:
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

    def _fd_jacobian(x_val, Fx_val, h=1e-5):
        """Compute Jacobian via forward differences. Uses 2 calls (Fx already known)."""
        nonlocal call_count
        J = np.zeros((2, 2))
        for j in range(2):
            if not _time_ok() or call_count >= 493:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x_val + e)
            if Fx_plus is not None:
                J[:, j] = (Fx_plus - Fx_val) / h
            else:
                J[:, j] = 1.0
        return J

    def _g(x_val):
        """Objective: 0.5 * ||F(x)||^2"""
        Fx = _call_F(x_val)
        if Fx is None:
            return None
        return 0.5 * np.dot(Fx, Fx)

    def _grad_g(x_val):
        """Gradient of g(x) = J(x)^T F(x). Uses 3 calls (2 for J, 1 for F)."""
        nonlocal call_count
        Fx = _call_F(x_val)
        if Fx is None:
            return None
        J = _fd_jacobian(x_val, Fx)
        # Gradient is J.T @ Fx
        grad = J.T @ Fx
        return grad

    def _scipy_fallback(x_start, max_iter=50):
        """
        Use scipy.optimize.minimize with dogleg method on g(x) = 0.5*||F(x)||^2.
        """
        nonlocal call_count
        
        # Wrap the objective and gradient to handle None returns safely
        def fun(x):
            val = _g(x)
            if val is None:
                return 1e10 # Large penalty
            return val
        
        def jac(x):
            val = _grad_g(x)
            if val is None:
                return np.array([1.0, 1.0]) # Fallback gradient
            return val

        try:
            res = minimize(fun, x_start, method='dogleg', jac=jac, options={'maxiter': max_iter, 'gtol': 1e-12})
            if res.success or (res.fun < best_norm and np.isfinite(res.fun)):
                # res.fun is 0.5 * ||F||^2, so ||F|| = sqrt(2 * res.fun)
                norm_est = np.sqrt(2 * res.fun) if res.fun > 0 else 0.0
                if norm_est < best_norm:
                    best_x[:] = res.x
                    best_norm = norm_est
                return res.x
        except Exception:
            pass
        return x_start

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian
    J = _fd_jacobian(x, Fx)
    try:
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-10:
            J += np.eye(2) * 1e-6
    except np.linalg.LinAlgError:
        J = np.eye(2)

    # Phase 1: Broyden's method main loop
    max_iter = 40
    stalled_count = 0
    prev_norm = norm_Fx
    use_fallback = False

    for it in range(max_iter):
        if not _time_ok() or call_count >= 485:
            break

        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                break

        norm_dx = np.linalg.norm(dx)
        if norm_dx == 0:
            break

        # Damped step: try full step, back off if needed
        alpha = 1.0
        success = False
        for _ in range(12):
            if not _time_ok() or call_count >= 488:
                break
            x_new = x + alpha * dx
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue
            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break
            norm_new = np.linalg.norm(Fx_new)
            if norm_new < norm_Fx * (1 + 1e-4):
                success = True
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break

        if not success:
            # Try smaller fixed step
            x_new = x + 0.1 * dx
            if np.all(np.isfinite(x_new)):
                Fx_new = _call_F(x_new)
                if Fx_new is not None:
                    norm_new = np.linalg.norm(Fx_new)
                    if norm_new < best_norm:
                        best_x = x_new.copy()
                        best_norm = norm_new
                    if norm_new <= 1e-8:
                        return x_new
                    x, Fx, norm_Fx = x_new, Fx_new, norm_new
            # If we couldn't even take a small step, or if we've been stuck, trigger fallback
            use_fallback = True
            break

        # Update Broyden Jacobian (good variant)
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        if s_norm_sq > 1e-18:
            J = J + np.outer(y - J @ s, s) / s_norm_sq

        x = x_new
        Fx = Fx_new
        norm_Fx = norm_new

        if norm_Fx <= 1e-8:
            return x

        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Detect stalling
        if norm_Fx > prev_norm * 0.999:
            stalled_count += 1
        else:
            stalled_count = 0
        prev_norm = norm_Fx

        if stalled_count >= 5:
            # Switch to robust fallback earlier if stalled
            use_fallback = True
            break

    # Phase 2: Robust Fallback if Broyden didn't solve it
    if norm_Fx > 1e-4 and use_fallback and _time_ok() and call_count < 480:
        x_fallback = _scipy_fallback(x, max_iter=30)
        Fx_fb = _call_F(x_fallback)
        if Fx_fb is not None:
            norm_fb = np.linalg.norm(Fx_fb)
            if norm_fb < best_norm:
                best_x = x_fallback.copy()
                best_norm = norm_fb
                x = x_fallback
                Fx = Fx_fb
                norm_Fx = norm_fb

    # Final polish: use FD-Newton for high precision if we're close
    if best_norm < 1e-4 and call_count < 495 and _time_ok():
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            norm_Fx = np.linalg.norm(Fx)
            for _ in range(10):
                if not _time_ok() or call_count >= 495:
                    break
                if norm_Fx <= 1e-8:
                    break
                # Small FD Jacobian
                h = max(1e-7, 1e-5 * np.sqrt(norm_Fx))
                J = _fd_jacobian(x, Fx, h)
                try:
                    if abs(np.linalg.det(J)) < 1e-12:
                        J += np.eye(2) * 1e-8
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                norm_dx = np.linalg.norm(dx)
                if norm_dx < 1e-14:
                    break
                # Damped step
                alpha = 1.0
                for _ in range(8):
                    if not _time_ok() or call_count >= 495:
                        break
                    x_new = x + alpha * dx
                    if not np.all(np.isfinite(x_new)):
                        alpha *= 0.5
                        continue
                    Fx_new = _call_F(x_new)
                    if Fx_new is None:
                        break
                    norm_new = np.linalg.norm(Fx_new)
                    if norm_new < norm_Fx:
                        x, Fx, norm_Fx = x_new, Fx_new, norm_new
                        if norm_Fx < best_norm:
                            best_x = x.copy()
                            best_norm = norm_Fx
                        break
                    alpha *= 0.5
                if norm_Fx <= 1e-8:
                    break

    # Final verification
    if call_count < 498 and _time_ok():
        Fx_final = _call_F(best_x)
        if Fx_final is not None:
            nf = np.linalg.norm(Fx_final)
            if nf <= 1e-8:
                return best_x
            if nf < best_norm:
                best_x = best_x.copy()
                best_norm = nf

    return best_x
# EVOLVE-BLOCK-END
