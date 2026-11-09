"""Root finder for 3 fixed two-dimensional systems using Broyden's method with call optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.

    Strategy:
    1. Broyden's method (1 F call/iter after setup) with multiple restarts.
    2. Initial Jacobian via forward differences (3 calls).
    3. Polish phase with Newton (forward when close, central when not).
    4. Final high-precision polish pass (tighter tolerance, central differences)
       to push below 1e-8 reliably when we are near the root.
    5. Strict call counting to stay under 500 and maximize score.
    6. Time-guarded with safety margin.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill

    x0 = np.array(x0, dtype=float)

    # Call tracking
    call_count = 0
    MAX_CALLS = 460  # under 500 to guarantee success if converged

    def time_ok():
        return time.time() < deadline

    def F_counted(x):
        """Evaluate F with call counting and safety checks. Returns None on failure."""
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
        except Exception:
            return None

    best_x = x0.copy()
    best_norm = np.inf

    def update_best(x, fx):
        """Update best solution if x is better. Never counts an extra call."""
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
                best_x = np.asarray(x, dtype=float).copy()
        except Exception:
            pass

    def get_jacobian_forward(x, Fx, h=None):
        """Forward-difference Jacobian. Requires 2 additional F calls."""
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
        """Central-difference Jacobian. Requires 4 additional F calls."""
        if h is None:
            h = 1e-7 * max(1.0, np.linalg.norm(x))
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx_p1 = F_counted(x + e1)
        Fx_m1 = F_counted(x - e1)
        Fx_p2 = F_counted(x + e2)
        Fx_m2 = F_counted(x - e2)
        if None in (Fx_p1, Fx_m1, Fx_p2, Fx_m2):
            return None
        col1 = (Fx_p1 - Fx_m1) / (2.0 * h)
        col2 = (Fx_p2 - Fx_m2) / (2.0 * h)
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

        if abs(np.linalg.det(J)) < 1e-10:
            J = J + 1e-8 * np.eye(2)

        prev_norm = n
        stagnation_count = 0

        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break

            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
            elif step_norm < 1e-15:
                break

            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
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

            if n_new >= prev_norm * 0.95:
                stagnation_count += 1
                if stagnation_count >= 3:
                    break
            else:
                stagnation_count = 0
            prev_norm = n_new

            s = x_new - x
            y = Fx_new - Fx
            denom = np.dot(s, s)
            if denom > 1e-30:
                J = J + np.outer(y - J @ s, s) / denom

            x = x_new
            Fx = Fx_new
            n = n_new

        return x, Fx, (n <= 1e-8)

    def polish(x_init, fx_init=None, max_iter=10, tol=1e-8):
        """
        Newton polishing for high precision.
        Uses central differences when close to the root, else forward.
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
            if call_count >= MAX_CALLS - 8 or not time_ok():
                break
            if n <= tol:
                return x, Fx, True

            # Choose difference scheme based on proximity to the root.
            if n < 1e-6:
                J = get_jacobian_central(x)
            else:
                J = get_jacobian_forward(x, Fx)

            if J is None:
                break

            if abs(np.linalg.det(J)) < 1e-10:
                J = J + 1e-8 * np.eye(2)

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
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
                dx_half = dx * 0.5
                x_half = x + dx_half
                if np.all(np.isfinite(x_half)):
                    Fx_half = F_counted(x_half)
                    if Fx_half is not None:
                        n_half = np.linalg.norm(Fx_half)
                        update_best(x_half, Fx_half)
                        if n_half < n:
                            x = x_half
                            Fx = Fx_half
                            n = n_half
                            continue
                break

        return x, Fx, (n <= tol)

    # Try Broyden from x0
    x_result, fx_result, success = broyden(x0)
    if success and best_norm <= 1e-8:
        return best_x

    # If close but not quite, try a standard polish
    if best_norm < 1e-4 and best_norm > 1e-8 and call_count < MAX_CALLS - 20 and time_ok():
        x_pol, fx_pol, pol_success = polish(best_x, None, max_iter=10, tol=1e-8)
        if pol_success and best_norm <= 1e-8:
            return best_x

    # Generate alternative starting points
    scale = max(1.0, np.linalg.norm(x0))
    candidates = []

    # Perturbations at various scales and directions
    for s in [0.5, 1.0, 2.0, 0.1, 5.0, 0.05]:
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]:
            candidates.append(x0 + s * scale * np.array([dx, dy]) * 0.5)

    # Scaled versions
    for factor in [0.5, 2.0, 0.1, 0.05, 5.0]:
        candidates.append(x0 * factor)

    # Origin and axis points
    candidates.append(np.array([0.0, 0.0]))
    candidates.append(np.array([scale, 0.0]))
    candidates.append(np.array([0.0, scale]))

    # Try Broyden from each candidate
    for x_init in candidates:
        if call_count >= MAX_CALLS - 20 or not time_ok():
            break
        if not np.all(np.isfinite(x_init)):
            continue
        x_result, fx_result, success = broyden(x_init)
        if success and best_norm <= 1e-8:
            return best_x

        # If close, try a standard polish
        if best_norm < 1e-4 and best_norm > 1e-8 and call_count < MAX_CALLS - 20 and time_ok():
            x_pol, fx_pol, pol_success = polish(best_x, None, max_iter=8, tol=1e-8)
            if pol_success and best_norm <= 1e-8:
                return best_x

    # Final high-precision polish pass: tighter tolerance, central differences,
    # more iterations. This is the refinement hypothesis: when we are near the
    # root, spend the remaining budget on an exact local solve to push below 1e-8.
    if best_norm > 1e-8 and best_norm < 1.0 and call_count < MAX_CALLS - 15 and time_ok():
        polish(best_x, None, max_iter=12, tol=1e-9)
        if best_norm <= 1e-8:
            return best_x

    # One more attempt: a couple of short high-precision polishes with a tighter tol.
    if best_norm > 1e-8 and call_count < MAX_CALLS - 10 and time_ok():
        for _ in range(3):
            if call_count >= MAX_CALLS - 5 or not time_ok():
                break
            if best_norm <= 1e-8:
                break
            polish(best_x, None, max_iter=4, tol=1e-9)
            if best_norm <= 1e-8:
                break

    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
