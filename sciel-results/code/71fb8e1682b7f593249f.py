"""Root finder for 3 fixed 2D systems using damped Gauss-Newton / Levenberg-Marquardt
with backtracking line search and Broyden Jacobian reuse."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 142.0  # safety margin before the 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def _finite2(v):
    try:
        v = np.asarray(v, dtype=float)
        return v.shape == (2,) and np.all(np.isfinite(v))
    except Exception:
        return False


def _fd_jac(F, x, h=1e-5):
    """Central-difference Jacobian (3 extra F-calls)."""
    J = np.empty((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2.0 * h)
    return J


def _solve_step(J, Fx, lam):
    """Return the step s = -J^-1 Fx, damping with lam if singular/ill-conditioned."""
    for L in (0.0, lam, lam * 100.0, lam * 1e6):
        A = J.T @ J
        if L > 0.0:
            A = A + L * np.eye(2)
        try:
            s = -np.linalg.solve(A, J.T @ Fx)
        except np.linalg.LinAlgError:
            continue
        if _finite2(s):
            return s
    return None


def _line_search(F, x, Fx, g, s, max_ls=30):
    """Backtracking line search; returns (x_new, F_new, g_new, steps)."""
    alpha = 1.0
    for _ in range(max_ls):
        xn = x + alpha * s
        Fxn = F(xn)
        if not _finite2(Fxn):
            alpha *= 0.5
            continue
        gn = 0.5 * float(np.dot(Fxn, Fxn))
        if gn <= g - 1e-4 * alpha * float(np.dot(s, JtF)) or gn < g:
            return xn, Fxn, gn, 1
        alpha *= 0.5
    # fallback: best trial
    return x, Fx, g, 0


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    if not _finite2(x):
        x = np.zeros(2)
    Fx = F(x)
    if not _finite2(Fx):
        return x
    g = 0.5 * float(np.dot(Fx, Fx))
    best_x, best_g = x.copy(), g
    tol = 1e-8
    lam = 1e-3
    J = _fd_jac(F, x)
    n_broyden = 0
    max_iter = 60

    for it in range(max_iter):
        if not _time_ok():
            break
        if g <= 1e-16:
            break

        # stale-model check: is the current J still a good linear model?
        # cheap test using last step residual; recompute if way off
        # (we do a light check by recomputing every 3 Broyden steps)
        if n_broyden >= 3:
            J = _fd_jac(F, x)
            n_broyden = 0

        s = _solve_step(J, Fx, lam)
        if s is None:
            # singular: recompute Jacobian and retry
            J = _fd_jac(F, x)
            n_broyden = 0
            s = _solve_step(J, Fx, lam)
            if s is None:
                break

        # line search
        alpha = 1.0
        step_taken = False
        for _ in range(30):
            xn = x + alpha * s
            Fxn = F(xn)
            if not _finite2(Fxn):
                alpha *= 0.5
                continue
            gn = 0.5 * float(np.dot(Fxn, Fxn))
            # sufficient decrease
            if gn <= g - 1e-4 * alpha * float(np.dot(s, J.T @ Fx)):
                step_taken = True
                break
            alpha *= 0.5
        if not step_taken:
            # no descent found; increase damping and retry
            lam *= 10.0
            if lam > 1e12:
                break
            continue

        # accept step
        # Broyden rank-1 update
        dF = Fxn - Fx
        dx = alpha * s
        denom = float(np.dot(dF, dx))
        if abs(denom) > 1e-30:
            J = J + (dF[:, None] - J @ dx[:, None]) / denom
        n_broyden += 1

        x, Fx, g = xn, Fxn, gn
        lam = max(1e-3, lam * 0.5)  # reduce damping on success

        if g < best_g:
            best_x, best_g = x.copy(), g
        if np.linalg.norm(Fx) <= tol:
            break

    return best_x
# EVOLVE-BLOCK-END
