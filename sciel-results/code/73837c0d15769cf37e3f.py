"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Broyden's method with damping and line search.
    Falls back to finite-difference Newton if Broyden stalls.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    x = np.array(x0, dtype=float).copy()
    Fx = np.array(F(x), dtype=float)
    norm_Fx = np.linalg.norm(Fx)

    if norm_Fx <= 1e-8:
        return x

    # Initialize Jacobian with finite differences (forward)
    J = _finite_diff_jacobian(F, x, Fx)

    best_x = x.copy()
    best_norm = norm_Fx

    # Broyden's method with damping
    for iteration in range(30):
        if time.time() > deadline:
            break

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Re-estimate Jacobian
            J = _finite_diff_jacobian(F, x, Fx)
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

        # Damped step with backtracking line search
        step = 1.0
        success = False
        for _ls in range(15):
            x_new = x + step * dx
            Fx_new = np.array(F(x_new), dtype=float)
            norm_new = np.linalg.norm(Fx_new)

            if not np.all(np.isfinite(Fx_new)):
                step *= 0.5
                continue

            if norm_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_new

            # Sufficient decrease
            if norm_new < (1 - 0.1 * step) * norm_Fx:
                # Broyden update
                s = x_new - x
                y = Fx_new - Fx
                # J = J + (y - J*s) * s^T / (s^T * s)
                Js = J @ s
                denom = np.dot(s, s)
                if denom > 1e-30:
                    J += np.outer(y - Js, s) / denom

                x = x_new
                Fx = Fx_new
                norm_Fx = norm_new
                success = True
                break
            else:
                step *= 0.5

        if success:
            if norm_Fx <= 1e-8:
                return x
        else:
            # If line search failed, try with a smaller fixed step
            x_new = x + 0.1 * dx
            Fx_new = np.array(F(x_new), dtype=float)
            if np.all(np.isfinite(Fx_new)):
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_new
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_new
            else:
                break

    # If we haven't converged, try a final polish with tighter tolerance
    if best_norm > 1e-8 and time.time() < deadline:
        x = best_x.copy()
        Fx = np.array(F(x), dtype=float)
        for _ in range(10):
            if time.time() > deadline:
                break
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-12:
                break
            J = _finite_diff_jacobian(F, x, Fx)
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            x_new = x + dx
            Fx_new = np.array(F(x_new), dtype=float)
            if np.all(np.isfinite(Fx_new)):
                x = x_new
                Fx = Fx_new
            else:
                break
        if np.linalg.norm(Fx) < best_norm:
            best_x = x.copy()
            best_norm = np.linalg.norm(Fx)

    return best_x


def _finite_diff_jacobian(F, x, Fx):
    """Compute Jacobian using forward differences."""
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (np.array(F(x + e), dtype=float) - Fx) / h
    return J
# EVOLVE-BLOCK-END
