"""Root finder for 3 fixed two-dimensional systems using robust Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with damping, then Newton polish for high precision.
    """
    x = np.array(x0, dtype=float, copy=True)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline or call_count >= 480:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
            result = np.asarray(result, dtype=float)
            if result.shape != (2,):
                result = result.reshape(2)
            if not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian via forward differences (2 extra calls)
    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        if Fx_plus is None:
            return best_x
        J[:, j] = (Fx_plus - Fx) / h

    # Ensure J is invertible
    det_J = np.linalg.det(J)
    if abs(det_J) < 1e-12:
        J += np.eye(2) * 1e-6

    # Broyden's Method with Damping
    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline or call_count >= 450:
            break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        # Damping strategy: try full step, then half, then quarter...
        alpha = 1.0
        accepted = False
        for _ in range(4):  # Max 4 damping attempts
            x_new = x + alpha * dx
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue

            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break

            norm_Fx_new = np.linalg.norm(Fx_new)

            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new

            if norm_Fx_new <= 1e-8:
                return x_new

            # Accept if we make significant progress
            if norm_Fx_new < norm_Fx * 0.9:
                # Update Jacobian using Broyden rank-1 formula
                s = x_new - x
                y = Fx_new - Fx
                s_norm_sq = np.dot(s, s)
                if s_norm_sq > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_norm_sq
                
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            break

    # If Broyden didn't reach 1e-8, try Newton polish with central differences
    if best_norm > 1e-8 and call_count < 470 and time.time() < deadline:
        x = best_x.copy()
        # Use central differences for more accurate Jacobian in polish phase
        h_polish = 1e-7
        J_new = np.zeros((2, 2))
        Fx_curr = _call_F(x)
        if Fx_curr is None:
            return best_x
        norm_Fx_curr = np.linalg.norm(Fx_curr)
        
        for j in range(2):
            e = np.zeros(2)
            e[j] = h_polish
            Fx_plus = _call_F(x + e)
            Fx_minus = _call_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                break
            J_new[:, j] = (Fx_plus - Fx_minus) / (2 * h_polish)
        else:
            # Central difference Jacobian computed successfully
            if abs(np.linalg.det(J_new)) > 1e-12:
                # Newton iterations for polish
                for _ in range(10):
                    if time.time() > deadline or call_count >= 475:
                        break
                    try:
                        dx = np.linalg.solve(J_new, -Fx_curr)
                    except np.linalg.LinAlgError:
                        break
                    
                    x_new = x + dx
                    if not np.all(np.isfinite(x_new)):
                        break
                    
                    Fx_new = _call_F(x_new)
                    if Fx_new is None:
                        break
                    
                    norm_Fx_new = np.linalg.norm(Fx_new)
                    
                    if norm_Fx_new < best_norm:
                        best_x = x_new.copy()
                        best_norm = norm_Fx_new
                    
                    if norm_Fx_new <= 1e-8:
                        return x_new
                    
                    # Update Jacobian using Broyden formula for Newton polish
                    s = x_new - x
                    y = Fx_new - Fx_curr
                    s_norm_sq = np.dot(s, s)
                    if s_norm_sq > 1e-16:
                        J_new = J_new + np.outer(y - J_new @ s, s) / s_norm_sq
                    
                    x = x_new
                    Fx_curr = Fx_new

    return best_x
# EVOLVE-BLOCK-END
