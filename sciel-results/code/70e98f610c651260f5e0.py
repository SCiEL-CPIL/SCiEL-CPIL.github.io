"""Root finder for 3 fixed 2D systems using derivative-free LM with rank-1 Jacobian updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 138.0
_t0 = None


def _elapsed():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return time.time() - _t0


def _time_ok():
    return _elapsed() < _DEADLINE


def _norm(x):
    return float(np.linalg.norm(x))


def _safe_F(F, x):
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _initial_jacobian(F, x, Fx):
    """Compute initial Jacobian via central differences (2 calls per column = 4 calls)."""
    h = 1e-7 * max(1.0, _norm(x))
    J = np.empty((2, 2), dtype=float)
    for j in range(2):
        e = np.zeros(2, dtype=float)
        e[j] = h
        y_plus = _safe_F(F, x + e)
        y_minus = _safe_F(F, x - e)
        if y_plus is None or y_minus is None:
            return None
        J[:, j] = (y_plus - y_minus) / (2 * h)
    return J


def _lm_solve(J, Fx, lam):
    """Solve (J^T J + lam*I) dx = -J^T Fx."""
    JTJ = J.T @ J
    A = JTJ + lam * np.eye(2)
    b = -J.T @ Fx
    try:
        dx = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            dx = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception:
            return None
    if not np.all(np.isfinite(dx)):
        return None
    return dx


def _rank1_update(J, dx, Fx_new, Fx_old):
    """Broyden rank-1 update: J_new = J + (Fx_new - Fx_old - J@dx) @ dx^T / (dx^T dx)."""
    s = dx
    y = Fx_new - Fx_old - J @ s
    denom = float(s @ s)
    if denom < 1e-30:
        return J
    return J + np.outer(y, s) / denom


def _lm_attempt(F, x0, max_iter=60):
    """Derivative-free LM with rank-1 Jacobian updates.
    Returns (best_x, best_norm, n_calls)."""
    x = np.array(x0, dtype=float)
    n_calls = 0
    best_x = x.copy()
    best_norm = np.inf

    Fx = _safe_F(F, x)
    if Fx is None:
        return x, np.inf, 1
    n_calls += 1
    nrm = _norm(Fx)
    if nrm < best_norm:
        best_norm, best_x = nrm, x.copy()
    if nrm <= 1e-12:
        return best_x, best_norm, n_calls

    # Initial Jacobian via central differences (4 calls)
    J = _initial_jacobian(F, x, Fx)
    if J is None:
        # Fallback: forward differences (2 calls)
        h = 1e-6 * max(1.0, _norm(x))
        J = np.empty((2, 2), dtype=float)
        ok = True
        for j in range(2):
            e = np.zeros(2, dtype=float)
            e[j] = h
            y = _safe_F(F, x + e)
            if y is None:
                ok = False
                break
            J[:, j] = (y - Fx) / h
            n_calls += 1
        if not ok:
            return best_x, best_norm, n_calls
    else:
        n_calls += 4

    lam = 1e-3  # initial damping factor
    prev_nrm = nrm

    for _ in range(max_iter):
        if not _time_ok():
            break

        # Solve for step
        dx = _lm_solve(J, Fx, lam)
        if dx is None:
            lam *= 10
            continue

        # Limit step size
        step_norm = _norm(dx)
        scale = max(1.0, _norm(x))
        if step_norm > 10 * scale:
            dx *= (10 * scale / step_norm)
            step_norm = _norm(dx)

        # Take step and evaluate
        x_new = x + dx
        Fx_new = _safe_F(F, x_new)
        n_calls += 1
        if Fx_new is None:
            lam *= 10
            continue

        nrm_new = _norm(Fx_new)

        # Check if improvement
        if nrm_new < nrm:
            # Accept step
            x = x_new
            Fx = Fx_new
            nrm = nrm_new
            if nrm < best_norm:
                best_norm, best_x = nrm, x.copy()
            if nrm <= 1e-12:
                return best_x, best_norm, n_calls

            # Update Jacobian via rank-1
            J = _rank1_update(J, dx, Fx_new, Fx - J @ dx)

            # Decrease lambda (improvement)
            lam = max(1e-10, lam * 0.5)
            prev_nrm = nrm
        else:
            # Reject step, increase lambda
            lam *= 10
            if lam > 1e15:
                break

    return best_x, best_norm, n_calls


def _powell_attempt(F, x0, max_iter=40):
    """Simple Powell-like coordinate descent as fallback.
    Returns (best_x, best_norm, n_calls)."""
    x = np.array(x0, dtype=float)
    n_calls = 0
    best_x = x.copy()
    best_norm = np.inf

    Fx = _safe_F(F, x)
    if Fx is None:
        return x, np.inf, 1
    n_calls += 1
    nrm = _norm(Fx)
    if nrm < best_norm:
        best_norm, best_x = nrm, x.copy()
    if nrm <= 1e-12:
        return best_x, best_norm, n_calls

    # Try small perturbations in each direction
    for scale in [1e-2, 1e-3, 1e-4, 1e-1]:
        if not _time_ok():
            break
        for j in range(2):
            for sign in [1, -1]:
                if not _time_ok():
                    break
                x_try = x.copy()
                x_try[j] += sign * scale
                Fx_try = _safe_F(F, x_try)
                n_calls += 1
                if Fx_try is None:
                    continue
                nrm_try = _norm(Fx_try)
                if nrm_try < nrm:
                    x = x_try
                    Fx = Fx_try
                    nrm = nrm_try
                    if nrm < best_norm:
                        best_norm, best_x = nrm, x.copy()
                    if nrm <= 1e-12:
                        return best_x, best_norm, n_calls

    return best_x, best_norm, n_calls


def find_root_system(F, x0):
    global _t0
    if _t0 is None:
        _t0 = time.time()

    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (2,):
        x0 = np.array([float(x0[0]), float(x0[1])]) if len(x0) >= 2 else np.array([0.0, 0.0])

    best_x = x0.copy()
    best_norm = np.inf
    total_calls = 0

    # Primary: LM with rank-1 updates from x0
    x1, n1, c1 = _lm_attempt(F, x0)
    total_calls += c1
    if n1 < best_norm:
        best_norm, best_x = n1, x1
    if best_norm <= 1e-12 or not _time_ok():
        return best_x

    # If LM didn't converge well, try a few perturbed starts
    if best_norm > 1e-6:
        for scale_p in [1e-2, 1e-3, 1e-1]:
            if not _time_ok():
                break
            for dx in [np.array([scale_p, 0.0]), np.array([-scale_p, 0.0]),
                       np.array([0.0, scale_p]), np.array([0.0, -scale_p])]:
                if not _time_ok():
                    break
                x_try = x0 + dx
                x2, n2, c2 = _lm_attempt(F, x_try, max_iter=30)
                total_calls += c2
                if n2 < best_norm:
                    best_norm, best_x = n2, x2
                if best_norm <= 1e-12:
                    return best_x

    # Last resort: coordinate descent
    if best_norm > 1e-6 and _time_ok():
        x3, n3, c3 = _powell_attempt(F, best_x, max_iter=20)
        total_calls += c3
        if n3 < best_norm:
            best_norm, best_x = n3, x3

    return best_x
# EVOLVE-BLOCK-END
