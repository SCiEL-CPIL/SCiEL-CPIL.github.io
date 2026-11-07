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
    max_iter = 60
    max_ls = 8

    # Levenberg-Marquardt parameter
    lambda_lm = 1e-8

    # Phase 1: LM/Newton hybrid until close to root
    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        # Compute Jacobian using forward differences (2 calls)
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h

        # Try to solve J dx = -Fx with LM regularization
        dx = None
        for _lm_iter in range(4):
            try:
                J_reg = J + lambda_lm * np.eye(2)
                dx = -np.linalg.solve(J_reg, Fx)
                break
            except np.linalg.LinAlgError:
                lambda_lm *= 10

        if dx is None:
            break

        # Check if we're close enough to switch to pure Newton polish
        # Pure Newton has quadratic convergence, so it's much faster near the root
        if norm_F < 1e-4:
            # Polish mode: pure Newton with minimal line search
            polish_success = False
            for _polish in range(10):
                if time.time() > deadline:
                    break

                # Recompute Jacobian for polish
                J_polish = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    J_polish[:, j] = (F(x + e) - Fx) / h

                try:
                    dx_polish = -np.linalg.solve(J_polish, Fx)
                except np.linalg.LinAlgError:
                    break

                # Try full step first, then halve if needed
                alpha = 1.0
                accepted = False
                for _ in range(max_ls):
                    x_candidate = x + alpha * dx_polish
                    F_candidate = F(x_candidate)
                    norm_candidate = np.linalg.norm(F_candidate)

                    if norm_candidate <= 1e-8:
                        return x_candidate

                    if norm_candidate < norm_F:
                        accepted = True
                        x = x_candidate
                        Fx = F_candidate
                        norm_F = norm_candidate
                        best_x = x.copy()
                        best_norm = norm_F
                        break
                    else:
                        alpha *= 0.5

                if not accepted:
                    break

                if norm_F < 1e-8:
                    return x

            if norm_F <= 1e-8:
                return x
            if norm_F < best_norm:
                best_x = x.copy()
                best_norm = norm_F
            # Continue to next iteration if not converged
            if norm_F >= 1e-8:
                continue

        # Normal LM/Newton line search
        alpha = 1.0
        step = dx
        accepted = False

        for _ in range(max_ls):
            x_candidate = x + alpha * step
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)

            if norm_candidate <= 1e-8:
                return x_candidate

            if norm_candidate < norm_F:
                accepted = True
                x = x_candidate
                Fx = F_candidate
                norm_F = norm_candidate
                best_x = x.copy()
                best_norm = norm_F
                break
            else:
                alpha *= 0.5

        if not accepted:
            # Increase LM parameter to make steps smaller
            lambda_lm *= 10
            if lambda_lm > 1e6:
                break
            continue

        # Decrease LM parameter if step was good
        if alpha >= 0.5:
            lambda_lm = max(lambda_lm * 0.5, 1e-10)

        if norm_F <= 1e-8:
            return x

    return best_x
# EVOLVE-BLOCK-END
