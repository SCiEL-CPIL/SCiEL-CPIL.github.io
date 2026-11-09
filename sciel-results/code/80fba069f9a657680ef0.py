"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 with ||F(x)|| <= 1e-8 using Broyden's method.
    Broyden's method uses 1 F-call per iteration after initial Jacobian setup,
    giving a significant call-count advantage over finite-difference Newton.
    """
    deadline = time.time() + 115  # 5s safety margin before 120s kill

    def _norm(v):
        return np.linalg.norm(v)

    def _safe_F(x):
        try:
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                raise ValueError("bad F output")
            return fx
        except Exception:
            return None

    def _broyden(F, x0, max_iter=100, tol=1e-10):
        """Broyden's good method with line search for 2D systems."""
        x = np.array(x0, dtype=float)
        Fx = _safe_F(x)
        if Fx is None:
            return None, 1
        if _norm(Fx) <= tol:
            return x, 1

        # Initial Jacobian via forward differences
        h = 1e-7
        J = np.zeros((2, 2))
        calls = 1
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = _safe_F(x + e)
            if Fx_e is None:
                return None, calls + 1
            calls += 1
            J[:, j] = (Fx_e - Fx) / h

        # Broyden iteration
        for it in range(max_iter):
            if time.time() > deadline:
                break

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]

            # Line search with backtracking
            alpha = 1.0
            Fx_norm = _norm(Fx)
            for _ in range(30):
                x_new = x + alpha * dx
                Fx_new = _safe_F(x_new)
                calls += 1
                if Fx_new is None:
                    alpha *= 0.5
                    continue
                Fx_new_norm = _norm(Fx_new)
                if Fx_new_norm <= Fx_norm * (1 - 1e-4 * alpha):
                    break
                alpha *= 0.5

            if Fx_new is None:
                break

            # Check convergence
            if Fx_new_norm <= tol:
                return x_new, calls

            # Broyden update: J_new = J + (s - J*dx) * v^T where v^T = (s - J*dx)^T / ||s - J*dx||^2
            s = x_new - x
            y = Fx_new - Fx
            Jdx = J @ dx
            r = s - Jdx
            r_norm_sq = np.dot(r, r)
            if r_norm_sq > 1e-20:
                J = J + np.outer(r, r) / r_norm_sq  # rank-1 update: J += r * r^T / ||r||^2

            x = x_new
            Fx = Fx_new

        return x, calls

    # Strategy 1: Broyden from x0
    x_best = None
    best_norm = np.inf
    total_calls = 0

    result, calls = _broyden(F, x0, max_iter=80, tol=1e-10)
    total_calls += calls
    if result is not None:
        Fx = _safe_F(result)
        if Fx is not None:
            n = _norm(Fx)
            if n < best_norm:
                best_norm = n
                x_best = result.copy()
            if n <= 1e-8:
                return x_best

    # Strategy 2: Try a few perturbed starting points if first didn't converge
    if best_norm > 1e-8:
        perturbations = [
            x0 + 0.1 * np.array([1.0, 0.0]),
            x0 + 0.1 * np.array([0.0, 1.0]),
            x0 + 0.1 * np.array([1.0, 1.0]) / np.sqrt(2),
            x0 - 0.1 * np.array([1.0, 0.0]),
            x0 + 0.5 * np.array([1.0, 0.0]),
            x0 + 0.5 * np.array([0.0, 1.0]),
        ]
        for xp in perturbations:
            if time.time() > deadline:
                break
            result, calls = _broyden(F, xp, max_iter=60, tol=1e-10)
            total_calls += calls
            if result is not None:
                Fx = _safe_F(result)
                if Fx is not None:
                    n = _norm(Fx)
                    if n < best_norm:
                        best_norm = n
                        x_best = result.copy()
                    if n <= 1e-8:
                        return x_best

    # Strategy 3: If still not converged, try a simple damped Newton with central differences
    # as a last resort (more expensive but more robust)
    if best_norm > 1e-8 and x_best is not None:
        x = x_best.copy()
        for _ in range(30):
            if time.time() > deadline:
                break
            Fx = _safe_F(x)
            if Fx is None:
                break
            n = _norm(Fx)
            if n <= 1e-8:
                return x

            # Central difference Jacobian (4 calls)
            h = 1e-7
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fp = _safe_F(x + e)
                Fm = _safe_F(x - e)
                total_calls += 2
                if Fp is None or Fm is None:
                    J = None
                    break
                J[:, j] = (Fp - Fm) / (2 * h)
            if J is None:
                break

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

            # Damped step
            alpha = 1.0
            for _ in range(20):
                x_new = x + alpha * dx
                Fx_new = _safe_F(x_new)
                total_calls += 1
                if Fx_new is None:
                    alpha *= 0.5
                    continue
                if _norm(Fx_new) < n:
                    break
                alpha *= 0.5

            if Fx_new is None:
                break
            x = x_new

            if _norm(Fx_new) < best_norm:
                best_norm = _norm(Fx_new)
                x_best = x.copy()
            if best_norm <= 1e-8:
                return x_best

    # Return best found
    if x_best is not None and np.all(np.isfinite(x_best)):
        return x_best

    # Last resort: return x0
    return np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
