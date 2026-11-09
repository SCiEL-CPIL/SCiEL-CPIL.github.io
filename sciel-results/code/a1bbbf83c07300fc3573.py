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


def _lm_step(J, Fx, lambda_val):
    """Compute Levenberg-Marquardt step: (J^T J + lambda * diag(J^T J)) s = -J^T Fx."""
    JT = J.T
    JTJ = JT @ J
    diag = np.diag(JTJ)
    # Ensure positive diagonal for damping
    diag = np.maximum(diag, 1e-15)
    damped = JTJ + lambda_val * np.diag(diag)
    rhs = -(JT @ Fx)
    try:
        s = np.linalg.solve(damped, rhs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        try:
            s = np.linalg.lstsq(damped, rhs, rcond=None)[0]
        except Exception:
            return None
    if s is None or not np.all(np.isfinite(s)):
        return None
    return s


def _newton_polish(F, x, Fx, fx_norm, max_iters=10):
    """Pure Newton polish for final convergence, using central differences."""
    for _ in range(max_iters):
        if not _time_ok():
            break
        if fx_norm <= 1e-12:
            break
        
        # Central difference Jacobian for higher accuracy
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fp, _ = _safe_F(F, x + e)
            Fm, _ = _safe_F(F, x - e)
            if Fp is None or Fm is None:
                break
            J[:, j] = (Fp - Fm) / (2 * h)
        else:
            # All finite
            if not np.all(np.isfinite(J)):
                break
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(s)):
                break
            
            x_new = x + s
            Fx_new, fx_new_norm = _safe_F(F, x_new)
            if Fx_new is None:
                break
            if fx_new_norm < fx_norm:
                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm
            else:
                break
    return x, fx_norm


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

    # Compute initial Jacobian with forward differences
    J = _compute_jacobian_fd(F, x, Fx, h=1e-6)
    if J is None:
        # Try with different step size
        J = _compute_jacobian_fd(F, x, Fx, h=1e-5)
    if J is None:
        J = _compute_jacobian_fd(F, x, Fx, h=1e-4)
    if J is None:
        return best_x

    # Levenberg-Marquardt with adaptive lambda
    lambda_val = 1e-3
    lambda_increase = 2.0
    lambda_decrease = 0.5
    max_lm_iters = 30

    for _ in range(max_lm_iters):
        if not _time_ok():
            break
        if fx_norm <= 1e-10:
            break

        # Compute LM step
        s = _lm_step(J, Fx, lambda_val)
        if s is None:
            # Try with larger lambda for more damping
            lambda_val *= lambda_increase
            if lambda_val > 1e10:
                break
            continue

        # Step size limit for stability
        s_norm = np.linalg.norm(s)
        if s_norm > 1e6:
            s = s / s_norm * 1e6

        x_new = x + s
        Fx_new, fx_new_norm = _safe_F(F, x_new)
        if Fx_new is None:
            lambda_val *= lambda_increase
            if lambda_val > 1e10:
                break
            continue

        if fx_new_norm < fx_norm:
            # Accept step, decrease lambda
            x = x_new
            Fx = Fx_new
            fx_norm = fx_new_norm
            lambda_val = max(lambda_val * lambda_decrease, 1e-12)
            
            if fx_norm < best_norm:
                best_norm = fx_norm
                best_x = x.copy()
            
            # Update Jacobian with Broyden's good method
            y = Fx - (Fx_new - Fx)  # This is just -y, let me fix
            # Actually: y = Fx_new - Fx
            y = Fx_new - Fx
            s_actual = x_new - x
            denom = s_actual @ s_actual
            if denom > 1e-15:
                J = J + np.outer(y - J @ s_actual, s_actual) / denom
            
            # Refresh Jacobian every few steps for accuracy
            # (handled by Broyden update above)
        else:
            # Reject step, increase lambda
            lambda_val *= lambda_increase
            if lambda_val > 1e10:
                break

    # Final Newton polish with central differences
    if fx_norm > 1e-10 and fx_norm < 1e-4:
        x, fx_norm = _newton_polish(F, x, Fx, fx_norm, max_iters=8)
        if fx_norm < best_norm:
            best_norm = fx_norm
            best_x = x.copy()

    return best_x
# EVOLVE-BLOCK-END
