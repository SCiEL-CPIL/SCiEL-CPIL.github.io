# EVOLVE-BLOCK-START
import numpy as np
import time


def _clean(x):
    """Ensure x is a finite array of shape (2,)."""
    x = np.asarray(x, dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)):
        return np.zeros(2)
    return x


def _safe_eval(F, x):
    """Safely evaluate F at x, returning (Fx, success)."""
    try:
        Fx = np.asarray(F(x), dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return None, False
        return Fx, True
    except Exception:
        return None, False


def _secant_solve(F, x0, deadline, max_iter=60):
    """
    2-point secant method for 2D systems.
    Uses two previous points to estimate the Jacobian via finite differences
    of the function values, then solves for the next step.
    """
    x = np.array(x0, dtype=float)
    Fx, ok = _safe_eval(F, x)
    if not ok:
        return x, 1, np.inf

    fx_norm = np.linalg.norm(Fx)
    if fx_norm <= 1e-12:
        return x, 1, fx_norm

    # Initialize second point with a small perturbation
    h = 1e-5
    x1 = x + np.array([h, 0.0])
    Fx1, ok1 = _safe_eval(F, x1)
    calls = 2
    if not ok1:
        x1 = x + np.array([0.0, h])
        Fx1, ok1 = _safe_eval(F, x1)
        calls += 1
        if not ok1:
            return x, calls, fx_norm

    best_x = x.copy()
    best_norm = fx_norm

    for _ in range(max_iter):
        if time.time() > deadline:
            break

        # Estimate Jacobian using the two points
        # J * (x1 - x) = Fx1 - Fx  =>  J = (Fx1 - Fx) / (x1 - x)
        # For 2D, we need two independent directions. Use the secant formula:
        # We have points (x, Fx) and (x1, Fx1). The secant Jacobian satisfies:
        # J @ (x1 - x) = Fx1 - Fx
        # This gives one constraint per row. For 2x2, we need another direction.
        # Use a third point for the second direction.
        h2 = 1e-5
        x2 = x + np.array([0.0, h2])
        Fx2, ok2 = _safe_eval(F, x2)
        calls += 1
        if not ok2:
            x2 = x + np.array([h2, h2])
            Fx2, ok2 = _safe_eval(F, x2)
            calls += 1
            if not ok2:
                break

        # Build Jacobian estimate: J @ [d1; d2] = [f1; f2]
        d1 = x1 - x
        d2 = x2 - x
        f1 = Fx1 - Fx
        f2 = Fx2 - Fx

        D = np.column_stack([d1, d2])  # 2x2 matrix
        F_diff = np.column_stack([f1, f2])  # 2x2 matrix
        # J @ D = F_diff  =>  J = F_diff @ D^{-1}
        try:
            J = F_diff @ np.linalg.inv(D)
        except np.linalg.LinAlgError:
            # Fallback: use identity or break
            J = np.eye(2)

        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(dx)):
            break

        # Damped step: try full step, then halve
        step = 1.0
        found_step = False
        for ls_iter in range(15):
            if time.time() > deadline:
                break
            x_new = x + step * dx
            if not np.all(np.isfinite(x_new)):
                step *= 0.5
                continue

            Fx_new, ok_new = _safe_eval(F, x_new)
            calls += 1
            if not ok_new:
                step *= 0.5
                continue

            fx_new_norm = np.linalg.norm(Fx_new)
            if fx_new_norm < best_norm:
                best_norm = fx_new_norm
                best_x = x_new.copy()

            if fx_new_norm < 0.5 * fx_norm or fx_new_norm <= 1e-12:
                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm
                found_step = True
                break
            else:
                step *= 0.5

        if fx_norm <= 1e-12:
            return x, calls, fx_norm

        if not found_step:
            # Stagnation: perturb and restart local secant
            x = best_x + np.random.normal(0, 1e-4, 2)
            Fx, ok = _safe_eval(F, x)
            calls += 1
            if not ok:
                break
            fx_norm = np.linalg.norm(Fx)
            if fx_norm < best_norm:
                best_norm = fx_norm
                best_x = x.copy()

    return best_x, calls, best_norm


def _broyden_solve(F, x0, deadline, max_iter=80):
    """
    Broyden's method with line search for 2D systems.
    """
    x = np.array(x0, dtype=float)
    Fx, ok = _safe_eval(F, x)
    if not ok:
        return x, 1, np.inf

    fx_norm = np.linalg.norm(Fx)
    if fx_norm <= 1e-12:
        return x, 1, fx_norm

    # Initialize Jacobian via forward differences
    h = 1e-7
    J = np.zeros((2, 2))
    calls = 1
    for j in range(2):
        ej = np.zeros(2)
        ej[j] = h
        Fx_pert, ok_pert = _safe_eval(F, x + ej)
        calls += 1
        if ok_pert:
            J[:, j] = (Fx_pert - Fx) / h
        else:
            J[:, j] = np.eye(2)[:, j]

    best_x = x.copy()
    best_norm = fx_norm

    for _ in range(max_iter):
        if time.time() > deadline:
            break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(dx)):
            break

        step = 1.0
        found_step = False
        for ls_iter in range(15):
            if time.time() > deadline:
                break
            x_new = x + step * dx
            if not np.all(np.isfinite(x_new)):
                step *= 0.5
                continue

            Fx_new, ok_new = _safe_eval(F, x_new)
            calls += 1
            if not ok_new:
                step *= 0.5
                continue

            fx_new_norm = np.linalg.norm(Fx_new)
            if fx_new_norm < best_norm:
                best_norm = fx_new_norm
                best_x = x_new.copy()

            if fx_new_norm < 0.5 * fx_norm or fx_new_norm <= 1e-12:
                s = x_new - x
                y = Fx_new - Fx
                s_norm_sq = np.dot(s, s)
                if s_norm_sq > 1e-28:
                    J = J + np.outer(y - J @ s, s) / s_norm_sq
                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm
                found_step = True
                break
            else:
                step *= 0.5

        if fx_norm <= 1e-12:
            return x, calls, fx_norm

        if not found_step:
            x = best_x + np.random.normal(0, 1e-4, 2)
            Fx, ok = _safe_eval(F, x)
            calls += 1
            if not ok:
                break
            fx_norm = np.linalg.norm(Fx)
            if fx_norm < best_norm:
                best_norm = fx_norm
                best_x = x.copy()
            # Reinitialize Jacobian
            for j in range(2):
                ej = np.zeros(2)
                ej[j] = h
                Fx_pert, ok_pert = _safe_eval(F, x + ej)
                calls += 1
                if ok_pert:
                    J[:, j] = (Fx_pert - Fx) / h
                else:
                    J[:, j] = np.eye(2)[:, j]

    return best_x, calls, best_norm


def _polish(F, x, deadline, max_iter=20):
    """
    High-precision Newton polish using central differences for the Jacobian.
    """
    x = np.array(x, dtype=float)
    Fx, ok = _safe_eval(F, x)
    if not ok:
        return x, 0, np.inf

    fx_norm = np.linalg.norm(Fx)
    if fx_norm <= 1e-12:
        return x, 1, fx_norm

    calls = 1
    h = 1e-8
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        # Central difference Jacobian
        J = np.zeros((2, 2))
        for j in range(2):
            ej = np.zeros(2)
            ej[j] = h
            Fx_p, ok_p = _safe_eval(F, x + ej)
            calls += 1
            Fx_m, ok_m = _safe_eval(F, x - ej)
            calls += 1
            if ok_p and ok_m:
                J[:, j] = (Fx_p - Fx_m) / (2 * h)
            elif ok_p:
                J[:, j] = (Fx_p - Fx) / h
            elif ok_m:
                J[:, j] = (Fx - Fx_m) / h
            else:
                break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(dx)):
            break

        x_new = x + dx
        Fx_new, ok_new = _safe_eval(F, x_new)
        calls += 1
        if not ok_new:
            break

        fx_new_norm = np.linalg.norm(Fx_new)
        if fx_new_norm < fx_norm:
            x = x_new
            Fx = Fx_new
            fx_norm = fx_new_norm
            if fx_norm <= 1e-12:
                break
        else:
            break

    return x, calls, fx_norm


def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using a multi-method strategy:
    1. Try Broyden's method (efficient, 1 call/step after init)
    2. If not converged, try secant method (derivative-free)
    3. Polish the best result with high-precision Newton
    4. Multi-start if needed
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    x0 = _clean(x0)

    # Strategy 1: Broyden's method
    x_best, calls_1, norm_1 = _broyden_solve(F, x0, deadline, max_iter=60)
    total_calls = calls_1

    if norm_1 <= 1e-8:
        # Polish for extra precision
        x_pol, calls_p, norm_p = _polish(F, x_best, deadline, max_iter=10)
        total_calls += calls_p
        if norm_p <= 1e-8:
            return _clean(x_pol)
        # Use whichever is better
        if norm_p < norm_1:
            x_best, norm_1 = x_pol, norm_p
        return _clean(x_best)

    # Strategy 2: Secant method from x0
    if time.time() < deadline - 10:
        x_sec, calls_2, norm_2 = _secant_solve(F, x0, deadline, max_iter=50)
        total_calls += calls_2
        if norm_2 < norm_1:
            x_best, norm_1 = x_sec, norm_2

    # Strategy 3: Multi-start Broyden from perturbed points
    if time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            x_trial = x0 + np.random.normal(0, 0.1, 2)
            x_m, calls_m, norm_m = _broyden_solve(F, x_trial, deadline, max_iter=40)
            total_calls += calls_m
            if norm_m < norm_1:
                x_best, norm_1 = x_m, norm_m
            if norm_1 <= 1e-8:
                break

    # Final polish
    if time.time() < deadline - 5:
        x_pol, calls_p, norm_p = _polish(F, x_best, deadline, max_iter=15)
        total_calls += calls_p
        if norm_p < norm_1:
            x_best = x_pol

    return _clean(x_best)
# EVOLVE-BLOCK-END
