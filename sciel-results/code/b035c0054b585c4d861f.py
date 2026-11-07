"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)

    if norm_F <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm_F

    # Adaptive finite difference step size
    h = 1e-5
    max_iter = 80
    max_ls = 10

    # Levenberg-Marquardt parameter
    lambda_lm = 1e-8

    # Phase 1: LM/Newton with adaptive line search
    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Compute Jacobian using forward differences
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h

        # Try to solve J dx = -Fx with LM regularization
        dx = None
        lam = lambda_lm
        for _lm_iter in range(8):
            try:
                J_reg = J + lam * np.eye(2)
                dx = -np.linalg.solve(J_reg, Fx)
                break
            except np.linalg.LinAlgError:
                lam *= 10.0

        if dx is None:
            # If we can't solve, try a steepest descent step
            dx = -Fx
            lam = 1.0

        # Line search with backtracking
        alpha = 1.0
        step = dx
        accepted = False
        x_new = x
        F_new = Fx
        norm_new = norm_F

        for _ in range(max_ls):
            if time.time() > deadline:
                break
            x_candidate = x + alpha * step
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)

            if norm_candidate <= 1e-8:
                return x_candidate

            # Armijo-like condition: accept if sufficient decrease
            if norm_candidate < norm_F * (1 - 1e-4 * alpha * np.dot(step, Fx) / max(norm_F**2, 1e-12)):
                accepted = True
                x_new = x_candidate
                F_new = F_candidate
                norm_new = norm_candidate
                break
            else:
                alpha *= 0.5

        if not accepted:
            # If no step improves, try a smaller step in the gradient direction
            grad_dir = -Fx
            alpha = 0.1
            for _ in range(5):
                if time.time() > deadline:
                    break
                x_candidate = x + alpha * grad_dir
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                if norm_candidate < norm_F:
                    accepted = True
                    x_new = x_candidate
                    F_new = F_candidate
                    norm_new = norm_candidate
                    break
                alpha *= 0.5

        if not accepted:
            # Stuck, try a small random perturbation
            for _ in range(10):
                if time.time() > deadline:
                    break
                perturbation = np.random.normal(0, 0.1, size=2)
                x_candidate = x + perturbation
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                if norm_candidate < norm_F:
                    accepted = True
                    x_new = x_candidate
                    F_new = F_candidate
                    norm_new = norm_candidate
                    break

        if not accepted:
            # If truly stuck, break
            break

        # Update
        x = x_new
        Fx = F_new
        norm_F = norm_new

        if norm_F < best_norm:
            best_x = x.copy()
            best_norm = norm_F

        # Adaptive h: reduce h when close to root for better Jacobian accuracy
        if norm_F < 1e-4:
            h = max(1e-7, h * 0.5)

        # Increase lambda if step was too large (rejection)
        if alpha < 0.1:
            lambda_lm *= 5.0
        else:
            lambda_lm = max(1e-10, lambda_lm * 0.5)

        # Check convergence
        if norm_F <= 1e-8:
            return x

    # Phase 2: Polish with smaller steps if needed
    if best_norm > 1e-8:
        x = best_x.copy()
        Fx = F(x)
        norm_F = np.linalg.norm(Fx)
        h = 1e-6
        lambda_lm = 1e-6

        for it in range(30):
            if time.time() > deadline:
                break

            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                J[:, j] = (F(x + e) - Fx) / h

            dx = None
            lam = lambda_lm
            for _lm_iter in range(8):
                try:
                    J_reg = J + lam * np.eye(2)
                    dx = -np.linalg.solve(J_reg, Fx)
                    break
                except np.linalg.LinAlgError:
                    lam *= 10.0

            if dx is None:
                break

            # Small line search
            alpha = 1.0
            for _ in range(5):
                if time.time() > deadline:
                    break
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)

                if norm_candidate <= 1e-8:
                    return x_candidate

                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    if norm_F < best_norm:
                        best_x = x.copy()
                        best_norm = norm_F
                    break
                else:
                    alpha *= 0.5

            if norm_F <= 1e-8:
                return x

    # Return the best result found
    return best_x
# EVOLVE-BLOCK-END
