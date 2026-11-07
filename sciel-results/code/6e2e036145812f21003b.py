"""Root finder for 3 fixed two-dimensional systems using Secant-Step Hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2.
    Uses a Secant-Step Hybrid approach:
    1. Broyden's method with 2-point secant Jacobian initialization.
    2. Periodic full finite-difference Jacobian recomputation (re-anchoring) every k iterations.
    3. Damped Newton with fresh Jacobian if Broyden stalls.
    4. Multi-start local refinement if still not converged.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    x = np.array(x0, dtype=float).copy()
    n = 2

    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    norm_Fx = float(np.linalg.norm(Fx))

    if not np.all(np.isfinite(Fx)):
        # If F(x0) is invalid, try a small perturbation
        for _ in range(5):
            x = x + np.random.normal(0, 0.1, n)
            Fx = np.array(F(x), dtype=float)
            if np.all(np.isfinite(Fx)):
                break
        norm_Fx = float(np.linalg.norm(Fx))
        if not np.all(np.isfinite(Fx)):
            return x  # best effort

    if norm_Fx <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm_Fx

    # Helper to compute Jacobian via central differences (4 calls)
    def compute_jacobian(x, h=1e-5):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = np.array(F(x + e), dtype=float)
            Fm = np.array(F(x - e), dtype=float)
            J[:, j] = (Fp - Fm) / (2 * h)
        return J

    # Helper to compute Jacobian via forward differences (2 calls)
    def compute_jacobian_fwd(x, h=1e-5):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = np.array(F(x + e), dtype=float)
            J[:, j] = (Fp - Fx) / h
        return J

    # Strategy 1: Broyden's method with secant initialization and re-anchoring
    # Initialize Jacobian via 2-point secant difference (2 calls)
    h_sec = 1e-5
    J = np.zeros((n, n))
    # Use forward secant for init to save calls
    for j in range(n):
        e = np.zeros(n)
        e[j] = h_sec
        Fp = np.array(F(x + e), dtype=float)
        J[:, j] = (Fp - Fx) / h_sec

    max_iter_broyden = 30
    reanchor_every = 5  # Recompute full Jacobian every 5 iterations

    for it in range(max_iter_broyden):
        if time.time() > deadline:
            break

        # Re-anchor: recompute full Jacobian every k iterations
        if it > 0 and it % reanchor_every == 0:
            J = compute_jacobian(x)

        # Try to solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Line search to ensure sufficient decrease
        alpha = 1.0
        max_ls = 10
        ls_success = False
        for _ in range(max_ls):
            x_new = x + alpha * dx
            Fx_new = np.array(F(x_new), dtype=float)
            norm_Fx_new = float(np.linalg.norm(Fx_new))

            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue

            if norm_Fx_new <= 1e-8:
                return x_new

            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new

            # Sufficient decrease condition
            if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                # Good-Broyden update
                s = x_new - x
                y = Fx_new - Fx
                s_dot_s = float(np.dot(s, s))
                if s_dot_s > 1e-30:
                    J += np.outer(y - J @ s, s) / s_dot_s
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                ls_success = True
                break

            alpha *= 0.5

        if not ls_success:
            break

        if norm_Fx <= 1e-8:
            return x

    # Strategy 2: Damped Newton with fresh Jacobian, multiple restarts
    if time.time() < deadline:
        for restart in range(6):
            if time.time() > deadline:
                break

            # Recompute Jacobian with central differences
            Fx = np.array(F(x), dtype=float)
            norm_Fx = float(np.linalg.norm(Fx))
            if not np.all(np.isfinite(Fx)):
                x = x + np.random.normal(0, 0.1, n)
                continue
            if norm_Fx <= 1e-8:
                return x
            if norm_Fx < best_norm:
                best_x = x.copy()
                best_norm = norm_Fx

            J = compute_jacobian(x)

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    x = x + np.random.normal(0, 0.1, n)
                    continue

            # Damped Newton step
            alpha = 1.0
            max_ls = 12
            newton_success = False
            for _ in range(max_ls):
                x_new = x + alpha * dx
                Fx_new = np.array(F(x_new), dtype=float)
                norm_Fx_new = float(np.linalg.norm(Fx_new))

                if not np.all(np.isfinite(Fx_new)):
                    alpha *= 0.5
                    continue

                if norm_Fx_new <= 1e-8:
                    return x_new

                if norm_Fx_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_Fx_new

                # Armijo condition for damping
                if norm_Fx_new < norm_Fx - 1e-4 * alpha * float(np.dot(dx, J.T @ dx)):
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    newton_success = True
                    break

                alpha *= 0.5

            if not newton_success:
                # If Newton step failed, perturb x to escape local issues
                x = x + np.random.normal(0, 0.05, n)

    # Strategy 3: Small grid search around best_x if still not converged
    if time.time() < deadline and best_norm > 1e-8:
        Fx_best = np.array(F(best_x), dtype=float)
        best_norm = float(np.linalg.norm(Fx_best))

        if best_norm > 1e-8:
            # Try a few offsets around best_x
            offsets = [
                [0.01, 0.0], [-0.01, 0.0], [0.0, 0.01], [0.0, -0.01],
                [0.001, 0.0], [-0.001, 0.0], [0.0, 0.001], [0.0, -0.001],
                [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1],
            ]
            for off in offsets:
                if time.time() > deadline:
                    break
                x_try = best_x + np.array(off)
                Fx_try = np.array(F(x_try), dtype=float)
                norm_try = float(np.linalg.norm(Fx_try))
                if norm_try <= 1e-8:
                    return x_try
                if norm_try < best_norm:
                    best_x = x_try
                    best_norm = norm_try

    return best_x
# EVOLVE-BLOCK-END
