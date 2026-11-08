"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _start_time) < _DEADLINE


def _safe_F(F, x):
    """Call F and return (Fx, fx_norm) or (None, inf) on failure."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return None, np.inf
        return Fx, np.linalg.norm(Fx)
    except Exception:
        return None, np.inf


def _compute_jacobian_fd(F, x, Fx, h=1e-6):
    """Compute Jacobian using forward differences. Returns J or None on failure."""
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp, _ = _safe_F(F, x + e)
        if Fp is None:
            return None
        J[:, j] = (Fp - Fx) / h
    if not np.all(np.isfinite(J)):
        return None
    return J


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    # Initial evaluation
    Fx, fx_norm = _safe_F(F, x)
    if Fx is None:
        return best_x

    if fx_norm < best_norm:
        best_norm = fx_norm
        best_x = x.copy()

    if fx_norm <= 1e-10:
        return x

    # Initialize Jacobian with forward differences
    J = _compute_jacobian_fd(F, x, Fx, h=1e-6)
    if J is None:
        return _fallback_newton(F, x, Fx, best_x, best_norm)

    # Broyden's Good Method
    max_iters = 50
    for _ in range(max_iters):
        if not _time_ok():
            break

        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Refresh Jacobian with FD and retry
            J_new = _compute_jacobian_fd(F, x, Fx, h=1e-5)
            if J_new is None:
                return _fallback_newton(F, x, Fx, best_x, best_norm)
            J = J_new
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                return _fallback_newton(F, x, Fx, best_x, best_norm)

        if not np.all(np.isfinite(s)):
            break

        # Line search with backtracking
        alpha = 1.0
        converged = False
        for _ in range(20):
            x_new = x + alpha * s
            Fx_new, fx_new_norm = _safe_F(F, x_new)
            if Fx_new is None:
                alpha *= 0.5
                continue

            if fx_new_norm < fx_norm:
                # Broyden update
                y = Fx_new - Fx
                s_actual = x_new - x
                denom = s_actual @ s_actual
                if denom > 1e-15:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom

                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm

                if fx_norm < best_norm:
                    best_norm = fx_norm
                    best_x = x.copy()

                if fx_norm <= 1e-10:
                    return x

                converged = True
                break
            alpha *= 0.5

        if not converged:
            break

    return best_x


def _fallback_newton(F, x, Fx, best_x, best_norm):
    """Fallback to standard Newton with forward-difference Jacobian."""
    for _ in range(20):
        if not _time_ok():
            break

        J = _compute_jacobian_fd(F, x, Fx, h=1e-5)
        if J is None:
            return best_x

        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            return best_x

        if not np.all(np.isfinite(s)):
            return best_x

        alpha = 1.0
        converged = False
        for _ in range(20):
            x_new = x + alpha * s
            Fx_new, fx_new_norm = _safe_F(F, x_new)
            if Fx_new is None:
                alpha *= 0.5
                continue

            if fx_new_norm < np.linalg.norm(Fx):
                x = x_new
                Fx = Fx_new
                if fx_new_norm < best_norm:
                    best_norm = fx_new_norm
                    best_x = x.copy()
                if fx_new_norm <= 1e-10:
                    return x
                converged = True
                break
            alpha *= 0.5

        if not converged:
            break

    return best_x
# EVOLVE-BLOCK-END
