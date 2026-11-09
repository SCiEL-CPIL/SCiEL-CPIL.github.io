"""Root finder for 3 fixed two-dimensional systems using Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method.
    Uses a deadline guard to ensure we finish within the evaluator's time limit.
    """
    x = np.array(x0, dtype=float)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Initial function evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        return x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian via finite differences
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = F(x + e)
        J[:, j] = (Fx_e - Fx) / h

    # Broyden's method iterations
    max_iter = 200
    best_x = x
    best_norm = norm_Fx

    for it in range(max_iter):
        # Check deadline
        if time.time() > deadline:
            break

        # Try to solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, use pseudo-inverse or restart
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Fallback: take a small step in direction of -Fx
                dx = -0.1 * Fx / max(norm_Fx, 1e-12)

        # Line search to ensure decrease
        x_new = x + dx
        Fx_new = F(x_new)
        norm_Fx_new = np.linalg.norm(Fx_new)

        if not np.all(np.isfinite(Fx_new)):
            # Step was bad, try smaller step
            for alpha in [0.5, 0.25, 0.1]:
                x_try = x + alpha * dx
                Fx_try = F(x_try)
                if np.all(np.isfinite(Fx_try)):
                    norm_try = np.linalg.norm(Fx_try)
                    if norm_try < best_norm:
                        x_new = x_try
                        Fx_new = Fx_try
                        norm_Fx_new = norm_try
                        break
            else:
                # No valid step found, break
                break

        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        if norm_Fx_new <= 1e-8:
            return x_new

        # Update Jacobian using Broyden's formula
        s = x_new - x
        y = Fx_new - Fx
        # Broyden rank-1 update: J_new = J + (y - J*s) * s^T / (s^T * s)
        Js = J @ s
        y_minus_Js = y - Js
        sTs = s @ s
        if sTs > 1e-14:
            J = J + np.outer(y_minus_Js, s) / sTs

        x = x_new
        Fx = Fx_new
        norm_Fx = norm_Fx_new

    # Check if we achieved the target
    if best_norm <= 1e-8:
        return best_x

    # If Broyden didn't converge, try a few Newton steps from best_x
    x = best_x.copy()
    Fx = F(x)
    norm_Fx = np.linalg.norm(Fx)

    if norm_Fx > 1e-8 and time.time() < deadline:
        for it in range(30):
            if time.time() > deadline:
                break
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = 1e-7
                J[:, j] = (F(x + e) - Fx) / 1e-7
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            x_new = x + dx
            Fx_new = F(x_new)
            norm_new = np.linalg.norm(Fx_new)
            if np.all(np.isfinite(Fx_new)) and norm_new < norm_Fx:
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_new
                if norm_Fx <= 1e-8:
                    return x
            else:
                # Damped step
                for alpha in [0.5, 0.25, 0.1]:
                    x_try = x + alpha * dx
                    Fx_try = F(x_try)
                    norm_try = np.linalg.norm(Fx_try)
                    if np.all(np.isfinite(Fx_try)) and norm_try < norm_Fx:
                        x = x_try
                        Fx = Fx_try
                        norm_Fx = norm_try
                        if norm_Fx <= 1e-8:
                            return x
                        break

    return x
# EVOLVE-BLOCK-END
