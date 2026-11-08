"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using a robust hybrid approach:
    1. Try direct Newton with finite-difference Jacobian.
    2. If Newton stalls, use Broyden's method with rank-1 update.
    3. If still not converged, try damped Newton with backtracking line search.
    4. Time-guarded: return best result before deadline.
    
    Budget: <500 calls to F.
    """
    x = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 145.0  # safety margin before 150s kill
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        if call_count >= 498:
            return None
        call_count += 1
        try:
            result = F(x_val)
        except Exception:
            return None
        result = np.asarray(result, dtype=float)
        if result.shape != (2,) or not np.all(np.isfinite(result)):
            return None
        return result

    # --- Phase 0: Check if x0 is already a root ---
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # --- Phase 1: Newton with finite-difference Jacobian ---
    h = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = _call_F(x + e)
        if Fx_e is None:
            break
        J[:, j] = (Fx_e - Fx) / h

    newton_success = False
    for it in range(30):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break

        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Recompute Jacobian every 2 steps
        if it % 2 == 0:
            Fx_ref = _call_F(x)
            if Fx_ref is None:
                break
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = _call_F(x + e)
                if Fx_e is None:
                    break
                J[:, j] = (Fx_e - Fx_ref) / h

        # Newton step
        try:
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-12:
                break
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break

        norm_dx = np.linalg.norm(dx)
        if norm_dx < 1e-15:
            break

        # Damped step with backtracking
        step_scale = 1.0
        x_new = None
        for _ in range(10):
            x_trial = x - step_scale * dx
            if np.all(np.isfinite(x_trial)):
                Fx_trial = _call_F(x_trial)
                if Fx_trial is None:
                    break
                norm_trial = np.linalg.norm(Fx_trial)
                if norm_trial < norm_Fx * (1 - 1e-4 * step_scale):
                    x_new = x_trial
                    Fx = Fx_trial
                    norm_Fx = norm_trial
                    if norm_Fx <= 1e-8:
                        return x
                    if norm_Fx < best_norm:
                        best_x = x.copy()
                        best_norm = norm_Fx
                    break
            step_scale *= 0.5

        if x_new is None:
            # Try smaller step
            x_trial = x - 0.1 * dx
            if np.all(np.isfinite(x_trial)):
                Fx_trial = _call_F(x_trial)
                if Fx_trial is not None:
                    norm_trial = np.linalg.norm(Fx_trial)
                    if norm_trial < norm_Fx:
                        x = x_trial
                        Fx = Fx_trial
                        norm_Fx = norm_trial
                        if norm_Fx < best_norm:
                            best_x = x.copy()
                            best_norm = norm_Fx
                        continue
            break

        x = x_new

    # --- Phase 2: Broyden's method ---
    if time.time() < deadline and call_count < 480:
        # Ensure we have a valid Jacobian
        if not np.all(np.isfinite(J)) or abs(np.linalg.det(J)) < 1e-12:
            Fx_ref = _call_F(x)
            if Fx_ref is None:
                return best_x
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = _call_F(x + e)
                if Fx_e is None:
                    break
                J[:, j] = (Fx_e - Fx_ref) / h

        Fx = _call_F(x)
        if Fx is None:
            return best_x
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        for it in range(150):
            if time.time() > deadline:
                break
            if call_count >= 495:
                break

            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.pinv(J) @ (-Fx)
                except Exception:
                    break

            norm_dx = np.linalg.norm(dx)
            if norm_dx < 1e-15:
                break

            # Damp if step is large
            if norm_dx > 5.0:
                dx = dx / norm_dx * 5.0

            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                dx = dx * 0.1
                x_new = x + dx
                if not np.all(np.isfinite(x_new)):
                    break

            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break
            norm_Fx_new = np.linalg.norm(Fx_new)

            if norm_Fx_new <= 1e-8:
                return x_new
            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new

            # Broyden rank-1 update
            s = x_new - x
            y = Fx_new - Fx
            norm_s = np.linalg.norm(s)
            if norm_s > 1e-14:
                J = J + np.outer(y - J @ s, s) / (s @ s)

            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new

    # --- Phase 3: Final check ---
    Fx_final = _call_F(x)
    if Fx_final is not None:
        norm_final = np.linalg.norm(Fx_final)
        if norm_final <= 1e-8:
            return x
        if norm_final < best_norm:
            best_x = x.copy()
            best_norm = norm_final

    return best_x
# EVOLVE-BLOCK-END
