"""Root finder for 3 fixed 2-D systems: robust quasi-Newton with safeguards."""
import time
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses robust Newton with line search, Broyden fallback, and deadline guard.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    tol = 1e-8
    best_x = np.array(x0, dtype=float)
    best_norm = float('inf')

    def time_left():
        return deadline - time.time()

    def is_finite(v):
        try:
            a = np.asarray(v, dtype=float)
            return a.shape == (2,) and np.all(np.isfinite(a))
        except Exception:
            return False

    def safe_F(x):
        try:
            fx = np.asarray(F(x), dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return None
            return fx
        except Exception:
            return None

    def update_best(x, Fx):
        nonlocal best_x, best_norm
        if is_finite(x):
            n = np.linalg.norm(Fx)
            if n < best_norm:
                best_norm = n
                best_x = np.array(x, dtype=float)

    # Strategy 1: Newton with finite-difference Jacobian and line search
    x = np.array(x0, dtype=float)
    max_newton_steps = 80

    for step in range(max_newton_steps):
        if time_left() < 1.0:
            break

        Fx = safe_F(x)
        if Fx is None:
            break
        update_best(x, Fx)
        if np.linalg.norm(Fx) <= tol:
            return x

        # Compute finite-difference Jacobian
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_F(x + e)
            if Fx_plus is None:
                break
            J[:, j] = (Fx_plus - Fx) / h
        else:
            # Try Newton step with line search
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = None

            if dx is not None and np.all(np.isfinite(dx)):
                # Backtracking line search
                alpha = 1.0
                fx_norm = np.linalg.norm(Fx)
                for _ in range(30):
                    x_trial = x + alpha * dx
                    Fx_trial = safe_F(x_trial)
                    if Fx_trial is None:
                        alpha *= 0.5
                        continue
                    fx_trial_norm = np.linalg.norm(Fx_trial)
                    update_best(x_trial, Fx_trial)
                    if fx_trial_norm <= tol:
                        return x_trial
                    if fx_trial_norm < fx_norm * (1 - 1e-4 * alpha):
                        x = x_trial
                        Fx = Fx_trial
                        break
                    alpha *= 0.5
            else:
                # Jacobian singular or step invalid; switch to Broyden
                break

        if np.linalg.norm(Fx) <= tol:
            return x

    # Strategy 2: Broyden's method (quasi-Newton)
    x = np.array(x0, dtype=float)
    B = np.eye(2)  # Initial Jacobian approximation
    max_broyden_steps = 80

    for step in range(max_broyden_steps):
        if time_left() < 1.0:
            break

        Fx = safe_F(x)
        if Fx is None:
            break
        update_best(x, Fx)
        if np.linalg.norm(Fx) <= tol:
            return x

        try:
            dx = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx  # fallback: steepest descent

        if not np.all(np.isfinite(dx)):
            dx = -Fx

        # Line search
        alpha = 1.0
        fx_norm = np.linalg.norm(Fx)
        for _ in range(30):
            x_trial = x + alpha * dx
            Fx_trial = safe_F(x_trial)
            if Fx_trial is None:
                alpha *= 0.5
                continue
            fx_trial_norm = np.linalg.norm(Fx_trial)
            update_best(x_trial, Fx_trial)
            if fx_trial_norm <= tol:
                return x_trial
            if fx_trial_norm < fx_norm * (1 - 1e-4 * alpha):
                # Update Broyden matrix
                s = x_trial - x
                y = Fx_trial - Fx
                denom = np.dot(y, s)
                if abs(denom) > 1e-12:
                    B = B + np.outer(y - B @ s, s) / denom
                x = x_trial
                Fx = Fx_trial
                break
            alpha *= 0.5

        if np.linalg.norm(Fx) <= tol:
            return x

    # Return best result found
    return best_x
# EVOLVE-BLOCK-END
