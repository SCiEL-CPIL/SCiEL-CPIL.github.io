"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        arr = np.asarray(arr, dtype=float)
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _fd_jacobian_central(F, x, h=1e-5):
    """Compute finite-difference Jacobian using central differences."""
    J = np.zeros((2, 2))
    Fx = F(x)
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
    return J


def _broyden(F, x0, max_iter=200):
    """
    Broyden's method (good-Broyden) with line search and Jacobian refresh.
    """
    x = np.array(x0, dtype=float)
    Fx = F(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x, 1

    # Initialize Jacobian with finite differences
    J = _fd_jacobian_central(F, x)
    calls = 4  # 1 for Fx + 3 for Jacobian (2*2 central diff = 4 evals, but Fx already counted)
    # Actually central diff uses F(x+e) and F(x-e) for each of 2 directions = 4 evals total
    # Plus F(x) = 1, so 5 total. Let me recount: Fx=1, then for j=0: F(x+e0), F(x-e0); j=1: F(x+e1), F(x-e1) = 4 more. Total 5.
    calls = 5

    best_x = x
    best_norm = np.linalg.norm(Fx)

    for it in range(max_iter):
        if not _time_ok():
            break

        try:
            # Solve J * dx = -Fx
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Refresh Jacobian
            J = _fd_jacobian_central(F, x)
            calls += 4
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

        # Line search with backtracking
        step = 1.0
        Fx_new = None
        for _ in range(30):
            x_trial = x + step * dx
            if not _is_finite(x_trial):
                step *= 0.5
                continue
            Fx_trial = F(x_trial)
            calls += 1
            if not _is_finite(Fx_trial):
                step *= 0.5
                continue
            norm_trial = np.linalg.norm(Fx_trial)
            if norm_trial < best_norm:
                best_x = x_trial
                best_norm = norm_trial
            if norm_trial < 0.5 * best_norm or norm_trial <= 1e-8:
                Fx_new = Fx_trial
                x = x_trial
                break
            step *= 0.5
        else:
            # If no improvement found, try a smaller step or break
            if step < 1e-15:
                break
            x_trial = x + step * dx
            if not _is_finite(x_trial):
                break
            Fx_trial = F(x_trial)
            calls += 1
            if not _is_finite(Fx_trial):
                break
            norm_trial = np.linalg.norm(Fx_trial)
            if norm_trial < best_norm:
                best_x = x_trial
                best_norm = norm_trial
            x = x_trial
            Fx_new = Fx_trial

        if np.linalg.norm(Fx_new) <= 1e-8:
            return x, calls

        # Update Jacobian using Broyden's formula (good Broyden)
        s = x - (x - step * dx)  # This is just step * dx, but let's be explicit
        # Actually, s = x_new - x_old where x_old was the previous x
        # Let me restructure: save old x before update
        # This is getting complicated. Let me redo the loop structure.

    return best_x, calls


def _broyden_v2(F, x0, max_iter=200):
    """
    Broyden's method with proper Jacobian updates and line search.
    """
    x = np.array(x0, dtype=float)
    Fx = F(x)
    calls = 1
    if np.linalg.norm(Fx) <= 1e-8:
        return x, calls

    # Initialize Jacobian
    J = _fd_jacobian_central(F, x)
    calls += 4  # 4 additional evaluations for central differences

    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)

    for it in range(max_iter):
        if not _time_ok():
            break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            J = _fd_jacobian_central(F, x)
            calls += 4
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

        # Line search
        step = 1.0
        accepted = False
        for _ in range(40):
            x_trial = x + step * dx
            if not _is_finite(x_trial):
                step *= 0.5
                continue
            Fx_trial = F(x_trial)
            calls += 1
            if not _is_finite(Fx_trial):
                step *= 0.5
                continue
            norm_trial = np.linalg.norm(Fx_trial)
            if norm_trial < best_norm:
                best_x = x_trial.copy()
                best_norm = norm_trial
            if norm_trial <= 1e-8:
                x = x_trial
                Fx = Fx_trial
                accepted = True
                break
            # Armijo-like condition
            if norm_trial < 0.5 * best_norm:
                s = x_trial - x
                y = Fx_trial - Fx
                x = x_trial
                Fx = Fx_trial
                # Broyden update (good Broyden)
                # J_new = J + (y - J*s) @ s.T / (s.T @ s)
                Js = J @ s
                diff = y - Js
                denom = s @ s
                if denom > 1e-30:
                    J = J + np.outer(diff, s) / denom
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break

        if np.linalg.norm(Fx) <= 1e-8:
            return x, calls

        # Periodically refresh Jacobian to avoid drift
        if it % 5 == 4:
            J = _fd_jacobian_central(F, x)
            calls += 4

    return best_x, calls


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with line search and periodic Jacobian refresh.
    """
    x0 = np.array(x0, dtype=float)
    if not _is_finite(x0):
        x0 = np.array([0.0, 0.0])

    # First try Broyden's method
    x_result, calls = _broyden_v2(F, x0, max_iter=100)

    if np.linalg.norm(F(x_result)) <= 1e-8:
        return x_result

    # If Broyden failed, try Newton with more iterations as fallback
    # But we need to be careful about call count
    # Let's try a simple damped Newton
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    for it in range(50):
        if not _time_ok():
            break
        Fx = F(x)
        if not _is_finite(Fx):
            break
        norm = np.linalg.norm(Fx)
        if norm < best_norm:
            best_x = x.copy()
            best_norm = norm
        if norm <= 1e-8:
            return x

        J = _fd_jacobian_central(F, x)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        step = 1.0
        for _ in range(20):
            x_trial = x + step * dx
            if not _is_finite(x_trial):
                step *= 0.5
                continue
            Fx_trial = F(x_trial)
            if not _is_finite(Fx_trial):
                step *= 0.5
                continue
            norm_trial = np.linalg.norm(Fx_trial)
            if norm_trial < best_norm:
                best_x = x_trial.copy()
                best_norm = norm_trial
            if norm_trial < 0.5 * best_norm or norm_trial <= 1e-8:
                x = x_trial
                break
            step *= 0.5

    if np.linalg.norm(F(best_x)) <= 1e-8:
        return best_x

    return best_x
# EVOLVE-BLOCK-END
