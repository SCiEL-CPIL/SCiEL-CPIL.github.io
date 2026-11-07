"""Root finder for 3 fixed 2D systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method (good for small n) with line search safeguards
    and a deadline guard to avoid timeouts.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    x = np.array(x0, dtype=float)
    if x.shape != (2,):
        x = x.reshape(2)

    # Initial function evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # Try a small perturbation
        x = x + np.array([1e-3, -1e-3])
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x

    # Compute initial Jacobian via finite differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-6
            Fx_pert = F(x + e)
            J[:, j] = (Fx_pert - Fx) / 1e-6
        return J

    J = compute_jacobian(x, Fx)

    # Broyden's method with line search
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    for iteration in range(100):
        if time.time() > deadline - 2.0:
            break

        # Check convergence
        if best_norm <= 1e-8:
            break

        # Solve for Newton step
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Fallback: use a damped step
            step = -Fx * 0.1

        # Line search: try full step, then damped steps
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-14:
            # Step is too small, try a small perturbation
            x = x + np.array([1e-6, -1e-6])
            Fx = F(x)
            if np.all(np.isfinite(Fx)):
                norm = np.linalg.norm(Fx)
                if norm < best_norm:
                    best_x = x.copy()
                    best_norm = norm
            continue

        alpha = 1.0
        found = False
        for _ in range(20):
            x_new = x + alpha * step
            Fx_new = F(x_new)
            if np.all(np.isfinite(Fx_new)):
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_new
                if norm_new < 0.5 * best_norm:
                    # Accept this step
                    # Update Jacobian using Broyden's rank-1 update
                    s = alpha * step
                    y = Fx_new - Fx
                    J = J + np.outer(y - J @ s, s) / (s @ s)
                    x = x_new
                    Fx = Fx_new
                    found = True
                    break
            alpha *= 0.5

        if not found:
            # No improvement found, try a smaller step or restart
            alpha = 0.1
            x_new = x + alpha * step
            Fx_new = F(x_new)
            if np.all(np.isfinite(Fx_new)):
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_new
                    s = alpha * step
                    y = Fx_new - Fx
                    J = J + np.outer(y - J @ s, s) / (s @ s)
                    x = x_new
                    Fx = Fx_new
                else:
                    # Stuck, try a random perturbation
                    x = best_x + np.random.randn(2) * 0.1
                    Fx = F(x)
                    if np.all(np.isfinite(Fx)):
                        best_norm = np.linalg.norm(Fx)
                        J = compute_jacobian(x, Fx)
                    else:
                        x = best_x
                        Fx = F(x)
                        J = compute_jacobian(x, Fx)
            else:
                x = best_x
                Fx = F(x)
                J = compute_jacobian(x, Fx)

        if best_norm <= 1e-8:
            break

    # Final check
    if np.all(np.isfinite(best_x)) and np.linalg.norm(F(best_x)) <= 1e-8:
        return best_x

    # If we didn't converge, return the best we found
    return best_x
# EVOLVE-BLOCK-END
