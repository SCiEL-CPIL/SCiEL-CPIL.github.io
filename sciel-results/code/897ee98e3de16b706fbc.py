"""Rewritten root-finder for 3 fixed 2D systems using a direct, efficient approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8.
    Strategy:
      1. Broyden's method (good initial Jacobian via finite differences, rank-1 updates).
      2. If stalled, try a damped Newton with adaptive step.
      3. Time-budgeted: return best before deadline.
    Budget: <500 calls. Each Broyden step = 1 call. Initial Jacobian = 3 calls.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
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
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            result = result.reshape(2)
        if not np.all(np.isfinite(result)):
            return None
        return result

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian via forward differences (2 calls: Fx already counted)
    h = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        if Fx_plus is None:
            break
        J[:, j] = (Fx_plus - Fx) / h
    else:
        # All Jacobian columns computed successfully
        pass
    
    # If Jacobian computation failed, use identity
    if J[:, 1].sum() == 0 and J[:, 0].sum() == 0:
        J = np.eye(2)
    else:
        # Check condition number; if ill-conditioned, regularize
        try:
            cond = np.linalg.cond(J)
            if cond > 1e10:
                J += np.eye(2) * 1e-8
        except np.linalg.LinAlgError:
            J = np.eye(2)

    # Broyden's method with adaptive damping
    stall_count = 0
    for it in range(480):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break

        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                break

        # Adaptive damping: start with full step, reduce if needed
        alpha = 1.0
        x_new = x
        Fx_new = None
        norm_Fx_new = np.inf
        
        for damp_iter in range(8):
            x_try = x + alpha * dx
            if not np.all(np.isfinite(x_try)):
                alpha *= 0.5
                continue
            
            Fx_try = _call_F(x_try)
            if Fx_try is None:
                alpha *= 0.5
                continue
            
            norm_Fx_try = np.linalg.norm(Fx_try)
            
            # Accept if we make progress, or if we've tried several dampings
            # (to allow Broyden updates even with slight increase)
            if norm_Fx_try < norm_Fx or damp_iter >= 3:
                x_new = x_try
                Fx_new = Fx_try
                norm_Fx_new = norm_Fx_try
                break
            else:
                alpha *= 0.5
        
        if Fx_new is None:
            break

        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        if norm_Fx_new <= 1e-8:
            return x_new

        # Broyden rank-1 update
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        if s_norm_sq > 1e-16:
            J = J + np.outer(y - J @ s, s) / s_norm_sq

        x = x_new
        Fx = Fx_new

        # Track stalling: use relative improvement threshold
        if norm_Fx_new >= best_norm * 0.99:
            stall_count += 1
        else:
            stall_count = 0

        # Restart with fresh Jacobian if stalled
        if stall_count >= 6:
            # Recompute Jacobian at current point
            h = 1e-5
            J_new = np.zeros((2, 2))
            success = True
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p = _call_F(x + e)
                if Fx_p is None:
                    success = False
                    break
                J_new[:, j] = (Fx_p - Fx) / h
            if success and abs(np.linalg.det(J_new)) > 1e-12:
                J = J_new
            stall_count = 0

    # Final polish: if close to root, do a few more Broyden steps with tighter tolerance
    if best_norm < 1e-4 and call_count < 490:
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx > 1e-8:
                # Recompute Jacobian for polish
                h = 1e-6
                J_polish = np.zeros((2, 2))
                success = True
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_p = _call_F(x + e)
                    if Fx_p is None:
                        success = False
                        break
                    J_polish[:, j] = (Fx_p - Fx) / h
                if success and abs(np.linalg.det(J_polish)) > 1e-12:
                    J = J_polish
                
                for polish_it in range(20):
                    if time.time() > deadline or call_count >= 495:
                        break
                    try:
                        dx = np.linalg.solve(J, -Fx)
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
                    
                    # Broyden update
                    s = x_new - x
                    y = Fx_new - Fx
                    s_norm_sq = np.dot(s, s)
                    if s_norm_sq > 1e-16:
                        J = J + np.outer(y - J @ s, s) / s_norm_sq
                    
                    x = x_new
                    Fx = Fx_new

    return best_x
# EVOLVE-BLOCK-END
