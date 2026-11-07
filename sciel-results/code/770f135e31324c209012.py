"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts and polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8.
    Uses Broyden's method (rank-1 Jacobian update) to minimize F-call count.
    Includes a final high-precision polish pass for near-converged solutions.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    tol = 1e-8
    start_time = time.time()
    time_limit = 135.0  # Safety margin for 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return time_limit - elapsed()

    best_x = x.copy()
    best_norm = np.linalg.norm(F(x))

    if best_norm <= tol:
        return best_x

    # Helper: compute finite-difference Jacobian (costs 2 F calls, reuses Fx)
    def fd_jacobian(x, Fx, h=1e-5):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_pert = F(x + e)
            J[:, j] = (Fx_pert - Fx) / h
        return J

    # Polish pass: high-precision Newton iterations using FD Jacobian
    def polish(x_start, max_iters=30):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local

        h = 1e-6 * max(1.0, np.linalg.norm(x_local))
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            if norm_Fx <= tol:
                return x_local

            # Compute fresh Jacobian each iteration for accuracy
            J = fd_jacobian(x_local, Fx, h)
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    J_reg = J + 1e-10 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Fx)
                except np.linalg.LinAlgError:
                    break

            # Take the step
            x_try = x_local + dx
            F_try = F(x_try)
            norm_try = np.linalg.norm(F_try)

            if norm_try < norm_Fx:
                x_local = x_try
                Fx = F_try
                norm_Fx = norm_try
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_local.copy()
            else:
                # Try half step
                x_try = x_local + 0.5 * dx
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)
                if norm_try < norm_Fx:
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_local.copy()
                else:
                    break

        return x_local

    # Broyden's method on a local basin
    def broyden_solve(x_start, max_iters=40):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local

        # Initial Jacobian via finite differences (2 F calls)
        J = fd_jacobian(x_local, Fx)

        for i in range(max_iters):
            if time_left() < 2.0:
                break

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    J_reg = J + 1e-8 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Fx)
                except np.linalg.LinAlgError:
                    break

            # Backtracking line search with any-decrease acceptance
            alpha = 1.0
            accepted = False
            for _ in range(10):
                if time_left() < 1.0:
                    break
                x_try = x_local + alpha * dx
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)
                if norm_try < norm_Fx:
                    # Update Jacobian (Broyden rank-1)
                    s = alpha * dx
                    y = F_try - Fx
                    Js = J @ s
                    denom = s @ s
                    if denom > 1e-14:
                        J += np.outer(y - Js, s) / denom
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    accepted = True
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_local.copy()
                    if norm_Fx <= tol:
                        return x_local
                    break
                else:
                    alpha *= 0.5

            if not accepted:
                break

        return x_local

    # Main loop with restarts
    current_x = x.copy()
    n_restarts = 0
    max_restarts = 25

    while n_restarts < max_restarts and time_left() > 5.0:
        if best_norm <= tol:
            return best_x

        # Try Broyden's method from current_x
        result = broyden_solve(current_x)
        if best_norm <= tol:
            return best_x

        n_restarts += 1

        # Choose new starting point
        if n_restarts % 4 == 0:
            # Periodically try points near x0
            current_x = x + np.random.normal(0, 1.0, size=2)
        elif n_restarts % 4 == 1:
            # Try around best_x with growing radius
            radius = min(10.0, 0.5 * n_restarts)
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec)
            current_x = best_x + radius * dir_vec
        elif n_restarts % 4 == 2:
            # Try in a wider box around x0
            current_x = x + np.random.uniform(-5, 5, size=2)
        else:
            # Try around best_x with moderate radius
            radius = min(5.0, 0.3 * n_restarts)
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec)
            current_x = best_x + radius * dir_vec

    # Final polish pass if we got close but didn't quite converge
    if best_norm < 1e-4 and time_left() > 5.0:
        polish(best_x, max_iters=40)

    return best_x
# EVOLVE-BLOCK-END
