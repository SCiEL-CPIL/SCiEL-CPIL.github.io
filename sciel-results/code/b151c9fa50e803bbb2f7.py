"""Root-finder for 3 fixed 2D systems: Efficient Broyden's method with smart restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).

    Strategy:
      1. Broyden's method (good-Broyden) with forward-difference initial Jacobian.
      2. Damped step acceptance to ensure progress.
      3. Smart restart: If Broyden stalls, recompute Jacobian via FD and continue.
      4. Strict call budget and time deadline management.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        if call_count >= 495:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        try:
            result = np.asarray(result, dtype=float)
        except Exception:
            return None
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _fd_jacobian(x, Fx):
        """Forward-difference Jacobian (2 extra calls, reuses Fx)."""
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x + e)
            if Fx_plus is None:
                return None
            J[:, j] = (Fx_plus - Fx) / h
        return J

    def _is_finite_2d(v):
        return np.all(np.isfinite(v))

    # ---- Initial evaluation ----
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # ---- Initial Jacobian via forward differences ----
    J = _fd_jacobian(x, Fx)
    if J is None:
        return best_x
    if abs(np.linalg.det(J)) < 1e-12:
        J = J + np.eye(2) * 1e-6

    # ---- Broyden's Method Loop ----
    stalled = 0
    max_stall = 3  # Restart after 3 stalled iterations
    
    for it in range(480):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break

        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, restart with new Jacobian
            J = _fd_jacobian(x, Fx)
            if J is None:
                break
            if abs(np.linalg.det(J)) < 1e-12:
                J = J + np.eye(2) * 1e-6
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

        if not _is_finite_2d(dx):
            break

        # Step control: limit step size to avoid divergence
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 5.0:
            dx = dx / norm_dx * 5.0

        # Try full step first, then damped steps
        accepted = False
        for damp in (1.0, 0.5, 0.25, 0.1, 0.05):
            x_new = x + damp * dx
            if not _is_finite_2d(x_new):
                continue
            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break
            norm_Fx_new = np.linalg.norm(Fx_new)
            if not _is_finite_2d(Fx_new):
                continue
            
            # Accept if residual decreases
            if norm_Fx_new < norm_Fx:
                # Broyden rank-1 update
                s = x_new - x
                y = Fx_new - Fx
                s_norm_sq = np.dot(s, s)
                if s_norm_sq > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_norm_sq
                
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                accepted = True
                
                if norm_Fx_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_Fx_new
                
                if norm_Fx_new <= 1e-8:
                    return x_new
                break
        
        if not accepted:
            # No step improved the residual.
            # Check if we are making any progress at all
            if norm_Fx < best_norm * 0.99:
                stalled = 0
            else:
                stalled += 1
            
            if stalled >= max_stall:
                # Restart: Recompute Jacobian from scratch
                J = _fd_jacobian(x, Fx)
                if J is None:
                    break
                if abs(np.linalg.det(J)) < 1e-12:
                    J = J + np.eye(2) * 1e-6
                stalled = 0
            else:
                # If we just failed to improve but haven't stalled completely,
                # it's better to break if we've used a lot of calls, 
                # but let's try one more restart logic if we are close.
                # Actually, if damp=1.0 failed to reduce, smaller steps might have.
                # The loop above tries them. If all fail, we are stuck.
                # Break to save calls? Or restart?
                # Let's restart immediately if we can't improve with damping.
                J = _fd_jacobian(x, Fx)
                if J is None:
                    break
                if abs(np.linalg.det(J)) < 1e-12:
                    J = J + np.eye(2) * 1e-6
                stalled = 0
                if call_count > 450:
                    break

    return best_x
# EVOLVE-BLOCK-END
