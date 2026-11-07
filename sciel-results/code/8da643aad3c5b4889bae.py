"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def time_left():
        return deadline - time.time()

    def norm_f(x):
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return np.inf
            return np.linalg.norm(Fx)
        except Exception:
            return np.inf

    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    # Initial Jacobian estimation using central differences for better accuracy
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        return J

    # Broyden's rank-1 method
    def broyden_iteration(x, Fx, J):
        """One Broyden iteration. Returns (new_x, new_Fx, new_J, converged)."""
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            try:
                step = np.linalg.pinv(J) @ Fx
            except Exception:
                return x, Fx, J, False

        new_x = x - step
        if not np.all(np.isfinite(new_x)):
            return x, Fx, J, False

        new_Fx = F(new_x)
        if not np.all(np.isfinite(new_Fx)):
            return x, Fx, J, False

        new_norm = np.linalg.norm(new_Fx)
        if new_norm <= 1e-12:
            return new_x, new_Fx, J, True

        # Broyden rank-1 update
        delta_x = new_x - x
        delta_F = new_Fx - Fx
        if np.linalg.norm(delta_F) > 1e-15:
            # s = delta_x, y = delta_F
            # J_new = J + (y - J*s) * s^T / (s^T * s)
            s = delta_x
            y = delta_F
            Js = J @ s
            correction = (y - Js) @ (s.T) / (s @ s)
            new_J = J + correction
        else:
            new_J = J

        return new_x, new_Fx, new_J, False

    # Main search loop with restarts
    max_restarts = 50
    max_iters_per_restart = 30

    for restart in range(max_restarts):
        if time_left() < 2.0:
            break

        if restart == 0:
            x = np.array(x0, dtype=float)
        else:
            # Perturb x for restart
            scale = 1.0 + restart * 0.5
            x = best_x + scale * np.random.randn(2) * 0.1

        # Compute initial F and J
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            continue
        Fx_norm = np.linalg.norm(Fx)
        if Fx_norm < best_norm:
            best_norm = Fx_norm
            best_x = x.copy()
        if Fx_norm <= 1e-12:
            return x

        J = compute_jacobian(x, Fx)

        for it in range(max_iters_per_restart):
            if time_left() < 2.0:
                break

            x, Fx, J, converged = broyden_iteration(x, Fx, J)
            if not np.all(np.isfinite(x)):
                break
            Fx_norm = np.linalg.norm(Fx)
            if Fx_norm < best_norm:
                best_norm = Fx_norm
                best_x = x.copy()
            if converged:
                return x

    # Final polish: if best_norm is close to target, do a few Newton steps
    if best_norm <= 1e-6:
        x = best_x.copy()
        for _ in range(10):
            if time_left() < 1.0:
                break
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                break
            Fx_norm = np.linalg.norm(Fx)
            if Fx_norm <= 1e-12:
                return x
            if Fx_norm < best_norm:
                best_norm = Fx_norm
                best_x = x.copy()
            J = compute_jacobian(x, Fx)
            try:
                step = np.linalg.solve(J, Fx)
                x = x - step
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(x)):
                break

    # Return the best x found
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
