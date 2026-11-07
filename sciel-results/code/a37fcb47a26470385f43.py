"""Root finder for 3 fixed 2D systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with a robust line search and a fallback to SciPy's hybr method.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    call_count = 0
    max_calls = 450  # Leave buffer for safety

    def F_wrapper(x):
        nonlocal call_count
        if call_count >= max_calls:
            return np.array([0.0, 0.0])  # Dummy to prevent crash, but we should stop
        call_count += 1
        try:
            val = F(x)
            val = np.asarray(val, dtype=float).flatten()
            if val.shape != (2):
                val = val.reshape(2)
            if not np.all(np.isfinite(val)):
                return np.array([1e10, 1e10])
            return val
        except Exception:
            return np.array([1e10, 1e10])

    x = np.array(x0, dtype=float).flatten()
    if x.shape != (2,):
        x = x.reshape(2)

    best_x = x.copy()
    best_norm = np.inf

    # Initial evaluation
    Fx = F_wrapper(x)
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx < best_norm:
        best_norm = norm_Fx
        best_x = x.copy()

    if norm_Fx <= 1e-8:
        return best_x

    # Compute initial Jacobian via finite differences (2 calls)
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = F_wrapper(x + e)
            J[:, j] = (Fx_plus - Fx) / h
        return J

    J = compute_jacobian(x, Fx)

    # Broyden's Method
    max_iter = 30
    for it in range(max_iter):
        if time.time() > deadline or call_count >= max_calls:
            break

        if norm_Fx <= 1e-8:
            return x

        # Solve J dx = -Fx
        try:
            # Add small regularization if J is ill-conditioned
            # Check condition number
            cond = np.linalg.cond(J)
            if cond > 1e12:
                J_reg = J + 1e-12 * np.eye(2)
                dx = np.linalg.solve(J_reg, -Fx)
            else:
                dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx = -Fx / max(np.linalg.norm(Fx), 1e-15)

        # Line Search (Backtracking)
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        accepted = False

        for _ in range(20):
            if time.time() > deadline or call_count >= max_calls:
                break

            x_new = x + alpha * dx
            Fx_new = F_wrapper(x_new)
            norm_Fx_new = np.linalg.norm(Fx_new)

            if norm_Fx_new < best_norm:
                best_norm = norm_Fx_new
                best_x = x_new.copy()

            # Armijo condition
            if norm_Fx_new <= norm_Fx - c1 * alpha * np.linalg.norm(dx) * norm_Fx:
                # Update Jacobian (Broyden)
                if np.linalg.norm(dx) > 1e-15:
                    delta_F = Fx_new - Fx
                    delta_x = x_new - x
                    # Broyden update
                    J = J + (delta_F[:, None] - J @ delta_x[:, None]) / np.dot(delta_x, delta_x)

                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                accepted = True
                break
            else:
                alpha *= rho

        if not accepted:
            # If line search fails, try a small step in negative gradient direction
            grad = J.T @ Fx
            if np.linalg.norm(grad) > 1e-15:
                step_size = 0.1 / np.linalg.norm(grad)
                x_new = x - step_size * grad
                Fx_new = F_wrapper(x_new)
                norm_Fx_new = np.linalg.norm(Fx_new)

                if norm_Fx_new < best_norm:
                    best_norm = norm_Fx_new
                    best_x = x_new.copy()

                if norm_Fx_new < norm_Fx:
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    # Update J roughly
                    if np.linalg.norm(x_new - x) > 1e-15:
                        delta_F = Fx_new - Fx
                        delta_x = x_new - x
                        J = J + (delta_F[:, None] - J @ delta_x[:, None]) / np.dot(delta_x, delta_x)
                else:
                    # Diverging, break and fallback
                    break

    # If Broyden didn't converge, try SciPy's hybr (Levenberg-Marquardt)
    if best_norm > 1e-8 and call_count < max_calls - 50:
        try:
            res = root(F_wrapper, x0, method='hybr')
            if res.success or np.linalg.norm(res.fun) < best_norm:
                x_res = res.x
                if np.all(np.isfinite(x_res)):
                    # Verify
                    F_res = F_wrapper(x_res)
                    norm_res = np.linalg.norm(F_res)
                    if norm_res < best_norm:
                        return x_res
                    if norm_res <= 1e-8:
                        return x_res
        except Exception:
            pass

    # Final polish: if we're close but not quite there, try a few more Newton steps
    if best_norm <= 1e-6 and call_count < max_calls - 10:
        x = best_x.copy()
        Fx = F_wrapper(x)
        norm_Fx = np.linalg.norm(Fx)
        for _ in range(5):
            if time.time() > deadline or call_count >= max_calls:
                break
            if norm_Fx <= 1e-8:
                return x
            try:
                J = compute_jacobian(x, Fx)
                dx = np.linalg.solve(J, -Fx)
                x_new = x + dx
                Fx_new = F_wrapper(x_new)
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new < best_norm:
                    best_norm = norm_Fx_new
                    best_x = x_new.copy()
                if norm_Fx_new < norm_Fx:
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
            except Exception:
                break

    return best_x
# EVOLVE-BLOCK-END
