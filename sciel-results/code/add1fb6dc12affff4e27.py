"""Root finder for 3 fixed two-dimensional systems: Broyden + trust-region + global fallback."""
import numpy as np
import time

_DEADLINE_FRAC = 0.88
_CALL_LIMIT = 490


def _elapsed_ok(start, limit_frac=_DEADLINE_FRAC, wall=120.0):
    return (time.time() - start) < limit_frac * wall


def _finite2(x):
    return (isinstance(x, np.ndarray) and x.shape == (2,) and np.all(np.isfinite(x)))


def _norm(v):
    return float(np.linalg.norm(v))


def _fd_jacobian(F, x, Fx, h=1e-5):
    """Forward-difference Jacobian. Returns (J, calls_used)."""
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_p = F(x + e)
        J[:, j] = (Fx_p - Fx) / h
    return J, 2


def _broyden_update(J, s, y):
    """Broyden's good rank-1 update."""
    s_norm_sq = np.dot(s, s)
    if s_norm_sq < 1e-30:
        return J
    # J += (y - J@s) @ s.T / (s.T @ s)
    diff = y - J @ s
    J = J + np.outer(diff, s) / s_norm_sq
    return J


def _solve(J, b):
    """Solve J x = b with fallback to least-squares."""
    try:
        return np.linalg.solve(J, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(J, b, rcond=None)[0]


def _trust_region_step(F, x, Fx, J, rho, calls_left, start):
    """
    Take a Newton step within trust region rho.
    Returns (x_new, Fx_new, used_calls, improved).
    """
    dx = _solve(J, Fx)
    if not np.all(np.isfinite(dx)):
        return x, Fx, 0, False

    dx_norm = _norm(dx)
    if dx_norm < 1e-15:
        return x, Fx, 0, False

    # Scale step to trust region
    if dx_norm > rho:
        dx = dx * (rho / dx_norm)
        dx_norm = rho

    f0 = 0.5 * _norm(Fx) ** 2
    alpha = 1.0
    c1 = 1e-4
    used = 0

    for _ in range(20):
        if used >= calls_left - 1 or not _elapsed_ok(start):
            break
        x_try = x + alpha * dx
        if not np.all(np.isfinite(x_try)):
            alpha *= 0.5
            continue
        Fx_try = F(x_try)
        used += 1
        if not np.all(np.isfinite(Fx_try)):
            alpha *= 0.5
            continue
        f_try = 0.5 * _norm(Fx_try) ** 2
        # Armijo condition
        if f_try <= f0 + c1 * alpha * np.dot(dx, Fx):
            return x_try, Fx_try, used, True
        alpha *= 0.5

    # If line search failed, try a small step in negative gradient direction of ||F||^2
    # Gradient of 0.5*||F||^2 is J^T F
    grad = J.T @ Fx
    if np.all(np.isfinite(grad)) and _norm(grad) > 1e-15:
        step_dir = -grad / _norm(grad)
        step_size = min(rho * 0.5, 0.1)
        x_try = x + step_size * step_dir
        if np.all(np.isfinite(x_try)):
            Fx_try = F(x_try)
            used += 1
            if np.all(np.isfinite(Fx_try)) and _norm(Fx_try) < _norm(Fx):
                return x_try, Fx_try, used, True

    return x, Fx, used, False


def _global_search(F, x, Fx, best_x, best_norm, calls_left, start):
    """
    Coarse grid / random search to find a better starting point.
    Returns (x_best, Fx_best, used_calls).
    """
    used = 0
    x_best = x.copy()
    Fx_best = Fx.copy()
    best_norm_local = best_norm

    scale = max(1.0, np.max(np.abs(x)))

    # Try a ring of points at various radii
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    n_angles = 8

    for r in radii:
        if used >= calls_left - 2 or not _elapsed_ok(start):
            break
        radius = scale * r
        for i in range(n_angles):
            if used >= calls_left - 1 or not _elapsed_ok(start):
                break
            angle = 2.0 * np.pi * i / n_angles
            x_try = x + radius * np.array([np.cos(angle), np.sin(angle)])
            if not np.all(np.isfinite(x_try)):
                continue
            try:
                Fx_try = F(x_try)
            except Exception:
                continue
            used += 1
            if not np.all(np.isfinite(Fx_try)):
                continue
            n = _norm(Fx_try)
            if n < best_norm_local:
                best_norm_local = n
                x_best = x_try.copy()
                Fx_best = Fx_try.copy()

    # Also try axis-aligned perturbations
    for axis in range(2):
        for sign in [-1, 1]:
            for r in [0.1, 0.5, 1.0, 5.0]:
                if used >= calls_left - 1 or not _elapsed_ok(start):
                    break
                x_try = x.copy()
                x_try[axis] += sign * scale * r
                if not np.all(np.isfinite(x_try)):
                    continue
                try:
                    Fx_try = F(x_try)
                except Exception:
                    continue
                used += 1
                if not np.all(np.isfinite(Fx_try)):
                    continue
                n = _norm(Fx_try)
                if n < best_norm_local:
                    best_norm_local = n
                    x_best = x_try.copy()
                    Fx_best = Fx_try.copy()

    return x_best, Fx_best, used


def find_root_system(F, x0):
    start = time.time()
    x = np.array(x0, dtype=float)

    # Initial evaluation
    try:
        Fx = np.asarray(F(x), dtype=float)
    except Exception:
        return x

    if not _finite2(Fx):
        return x

    best_x = x.copy()
    best_norm = _norm(Fx)
    calls = 1

    if best_norm <= 1e-8:
        return best_x

    # Initial Jacobian estimate
    J, j_calls = _fd_jacobian(F, x, Fx, h=1e-5)
    calls += j_calls

    if not np.all(np.isfinite(J)):
        # Try larger step
        J, j_calls = _fd_jacobian(F, x, Fx, h=1e-3)
        calls += j_calls
        if not np.all(np.isfinite(J)):
            J = np.eye(2)

    rho = max(1.0, np.max(np.abs(x)) * 0.5)
    rho_min = 1e-6

    max_iters = 60
    for it in range(max_iters):
        if calls >= _CALL_LIMIT or not _elapsed_ok(start):
            break

        # Check convergence
        if best_norm <= 1e-10:
            break

        # Take a trust-region step
        x_new, Fx_new, used, improved = _trust_region_step(
            F, x, Fx, J, rho, _CALL_LIMIT - calls, start
        )
        calls += used

        if not _elapsed_ok(start) or calls >= _CALL_LIMIT:
            break

        if improved:
            # Broyden update
            s = x_new - x
            y = Fx_new - Fx
            J = _broyden_update(J, s, y)

            n_new = _norm(Fx_new)
            if n_new < best_norm:
                best_x = x_new.copy()
                best_norm = n_new

            # Update trust region
            actual_reduction = 0.5 * (_norm(Fx) ** 2 - _norm(Fx_new) ** 2)
            if actual_reduction > 0.25 * 0.5 * _norm(Fx) ** 2:
                # Good progress: expand trust region
                rho = min(rho * 2.0, 10.0 * max(1.0, np.max(np.abs(x))))
            else:
                # Moderate progress: keep or slightly shrink
                rho = rho * 0.9

            x = x_new
            Fx = Fx_new
        else:
            # Step failed: shrink trust region
            rho *= 0.5
            if rho < rho_min:
                # Trust region too small; try global search
                x_g, Fx_g, used_g = _global_search(
                    F, x, Fx, best_x, best_norm, _CALL_LIMIT - calls, start
                )
                calls += used_g
                if calls >= _CALL_LIMIT or not _elapsed_ok(start):
                    break
                if _norm(Fx_g) < best_norm:
                    best_x = x_g.copy()
                    best_norm = _norm(Fx_g)
                    x = x_g.copy()
                    Fx = Fx_g.copy()
                    # Re-estimate Jacobian
                    J, j_calls = _fd_jacobian(F, x, Fx, h=1e-5)
                    calls += j_calls
                    if not np.all(np.isfinite(J)):
                        J = np.eye(2)
                    rho = max(1.0, np.max(np.abs(x)) * 0.5)
                    continue

                # Try a few more iterations with smaller rho before giving up on this point
                rho = rho_min

    # Final polish: if we're close, do a few more Newton steps
    if calls < _CALL_LIMIT - 10 and _elapsed_ok(start) and best_norm < 1e-4:
        x = best_x.copy()
        try:
            Fx = np.asarray(F(x), dtype=float)
            calls += 1
        except Exception:
            Fx = Fx  # keep old

        if _finite2(Fx):
            J, j_calls = _fd_jacobian(F, x, Fx, h=1e-6)
            calls += j_calls
            for _ in range(5):
                if calls >= _CALL_LIMIT - 1 or not _elapsed_ok(start):
                    break
                dx = _solve(J, Fx)
                if not np.all(np.isfinite(dx)) or _norm(dx) < 1e-15:
                    break
                x_try = x + dx
                if not np.all(np.isfinite(x_try)):
                    break
                Fx_try = F(x_try)
                calls += 1
                if not np.all(np.isfinite(Fx_try)):
                    break
                n_try = _norm(Fx_try)
                if n_try < _norm(Fx):
                    s = x_try - x
                    y = Fx_try - Fx
                    J = _broyden_update(J, s, y)
                    x = x_try
                    Fx = Fx_try
                    if n_try < best_norm:
                        best_x = x_try.copy()
                        best_norm = n_try
                else:
                    break
                if best_norm <= 1e-10:
                    break

    return best_x
