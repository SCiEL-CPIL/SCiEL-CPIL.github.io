"""Root finder for 3 fixed 2D systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with line search and fallback strategies.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    x = np.array(x0, dtype=float)
    if x.shape != (2,):
        x = x.reshape(2)

    # Initial function evaluation
    Fx = np.array(F(x), dtype=float)
    n_calls = 1
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    if best_norm <= 1e-8:
        return x

    # Broyden's method
    # Initialize Jacobian with identity
    J = np.eye(2)

    # Finite difference step size for initial Jacobian estimate
    h = 1e-6

    # Compute initial Jacobian via finite differences
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = np.array(F(x + e), dtype=float)
        n_calls += 1
        J[:, j] = (Fx_plus - Fx) / h

    # Broyden iteration
    for iteration in range(80):
        # Check deadline
        if time.time() > deadline - 5.0:
            break

        # Check convergence
        if best_norm <= 1e-8:
            return best_x

        # Solve J * s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use gradient descent step
            s = -Fx

        # Line search with Armijo condition
        alpha = 1.0
        c1 = 1e-4
        Fx_norm = np.linalg.norm(Fx)

        for ls_iter in range(30):
            x_new = x + alpha * s
            Fx_new = np.array(F(x_new), dtype=float)
            n_calls += 1

            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue

            Fx_new_norm = np.linalg.norm(Fx_new)

            # Armijo condition
            if Fx_new_norm <= Fx_norm - c1 * alpha * np.dot(s, Fx):
                # Check if this is better
                if Fx_new_norm < best_norm:
                    best_x = x_new.copy()
                    best_norm = Fx_new_norm

                # Update Jacobian using Broyden's formula
                if Fx_new_norm < best_norm:
                    Fx_new_norm_local = Fx_new_norm
                    s_actual = x_new - x
                    F_diff = Fx_new - Fx
                    # Broyden update: J_new = J + (F_diff - J*s_actual) * s_actual^T / (s_actual^T * s_actual)
                    J_s = J @ s_actual
                    r = F_diff - J_s
                    s_dot_s = np.dot(s_actual, s_actual)
                    if s_dot_s > 1e-15:
                        J = J + np.outer(r, s_actual) / s_dot_s

                x = x_new
                Fx = Fx_new
                break
            else:
                alpha *= 0.5

        else:
            # Line search failed, try a smaller step in negative gradient direction
            s = -Fx
            alpha = 0.1
            x_new = x + alpha * s
            Fx_new = np.array(F(x_new), dtype=float)
            n_calls += 1
            if np.all(np.isfinite(Fx_new)):
                Fx_new_norm = np.linalg.norm(Fx_new)
                if Fx_new_norm < best_norm:
                    best_x = x_new.copy()
                    best_norm = Fx_new_norm
                x = x_new
                Fx = Fx_new
                # Recompute Jacobian
                Fx = np.array(F(x), dtype=float)
                n_calls += 1
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_plus = np.array(F(x + e), dtype=float)
                    n_calls += 1
                    J[:, j] = (Fx_plus - Fx) / h

    # If Broyden didn't converge, try a few more strategies
    if best_norm > 1e-8 and time.time() < deadline - 10.0:
        # Try scipy's root with hybr method
        try:
            from scipy.optimize import root as scipy_root

            def F_wrapped(x):
                nonlocal n_calls
                if time.time() > deadline - 5.0:
                    return np.array([0.0, 0.0])
                result = np.array(F(x), dtype=float)
                n_calls += 1
                return result

            result = scipy_root(F_wrapped, x, method='hybr', tol=1e-10, maxfev=100)
            if result.success and np.all(np.isfinite(result.x)):
                Fx_result = np.array(F(result.x), dtype=float)
                n_calls += 1
                result_norm = np.linalg.norm(Fx_result)
                if result_norm < best_norm:
                    best_x = result.x.copy()
                    best_norm = result_norm
        except Exception:
            pass

    # Final check
    if best_norm <= 1e-8:
        return best_x

    # Return the best we found
    return best_x
# EVOLVE-BLOCK-END
