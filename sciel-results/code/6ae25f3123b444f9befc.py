"""Root finder for 3 fixed 2D systems: hybrid damped Newton + Broyden with Nelder-Mead fallback."""
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


def _fd_jac(F, x, h=1e-7):
    """Central-difference Jacobian (3 extra F-calls)."""
    J = np.empty((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2.0 * h)
    return J


def _newton_broyden(F, x0, max_iter=60, tol=1e-8):
    """Damped Gauss-Newton with Broyden updates and backtracking line search."""
    x = np.array(x0, dtype=float)
    Fx = F(x)
    if not _finite2(Fx):
        return x, np.inf
    g = 0.5 * float(np.dot(Fx, Fx))
    best_x, best_g = x.copy(), g
    lam = 1e-3
    J = _fd_jac(F, x)
    n_broyden = 0

    for it in range(max_iter):
        if not _time_ok():
            break
        if g <= 1e-16:
            break

        # Recompute Jacobian every 3 Broyden steps
        if n_broyden >= 3:
            J = _fd_jac(F, x)
            n_broyden = 0

        # Solve for step with Levenberg-Marquardt damping
        s = None
        for L in (0.0, lam, lam * 100.0, lam * 1e6):
            A = J.T @ J
            if L > 0.0:
                A = A + L * np.eye(2)
            try:
                s = -np.linalg.solve(A, J.T @ Fx)
            except np.linalg.LinAlgError:
                continue
            if _finite2(s):
                break

        if s is None:
            J = _fd_jac(F, x)
            n_broyden = 0
            for L in (0.0, lam, lam * 100.0, lam * 1e6):
                A = J.T @ J
                if L > 0.0:
                    A = A + L * np.eye(2)
                try:
                    s = -np.linalg.solve(A, J.T @ Fx)
                except np.linalg.LinAlgError:
                    continue
                if _finite2(s):
                    break
            if s is None:
                break

        # Backtracking line search
        alpha = 1.0
        step_taken = False
        JtF = J.T @ Fx
        for _ in range(30):
            xn = x + alpha * s
            Fxn = F(xn)
            if not _finite2(Fxn):
                alpha *= 0.5
                continue
            gn = 0.5 * float(np.dot(Fxn, Fxn))
            if gn <= g - 1e-4 * alpha * float(np.dot(s, JtF)) or gn < g:
                step_taken = True
                break
            alpha *= 0.5

        if not step_taken:
            lam *= 10.0
            if lam > 1e12:
                break
            continue

        # Accept step
        dF = Fxn - Fx
        dx = alpha * s
        denom = float(np.dot(dF, dx))
        if abs(denom) > 1e-30:
            J = J + (dF[:, None] - J @ dx[:, None]) / denom
        n_broyden += 1

        x, Fx, g = xn, Fxn, gn
        lam = max(1e-3, lam * 0.5)

        if g < best_g:
            best_x, best_g = x.copy(), g
        if np.linalg.norm(Fx) <= tol:
            break

    return best_x, best_g


def _nelder_mead(F, x0, max_iter=200, tol=1e-8):
    """Nelder-Mead minimization of ||F(x)||^2."""
    def objective(x):
        Fx = F(x)
        if not _finite2(Fx):
            return 1e300
        return float(np.dot(Fx, Fx))

    x = np.array(x0, dtype=float)
    simplex = [x.copy()]
    for i in range(2):
        xi = x.copy()
        xi[i] += 0.1 * (1.0 + abs(x[i]))
        simplex.append(xi)

    # Evaluate and sort
    fvals = [objective(v) for v in simplex]
    for i in range(2):
        for j in range(i, -1, -1):
            if fvals[i] < fvals[j]:
                simplex[i], simplex[j] = simplex[j], simplex[i]
                fvals[i], fvals[j] = fvals[j], fvals[i]

    best = simplex[0].copy()
    best_val = fvals[0]

    alpha, gamma, rho = 0.5, 2.0, 0.5

    for it in range(max_iter):
        if not _time_ok():
            break
        if best_val <= tol * tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)

        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < max(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, 3):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])

        # Re-sort
        for i in range(2):
            for j in range(i, -1, -1):
                if fvals[i] < fvals[j]:
                    simplex[i], simplex[j] = simplex[j], simplex[i]
                    fvals[i], fvals[j] = fvals[j], fvals[i]

        if fvals[0] < best_val:
            best = simplex[0].copy()
            best_val = fvals[0]

    return best, np.sqrt(best_val)


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    if not _finite2(x):
        x = np.zeros(2)

    # Phase 1: Newton-Broyden from x0
    x1, n1 = _newton_broyden(F, x, max_iter=60)
    if n1 <= 1e-16:
        return x1

    # Phase 2: Nelder-Mead from x0
    x2, n2 = _nelder_mead(F, x, max_iter=200)
    best_x, best_n = x1, n1
    if n2 < best_n:
        best_x, best_n = x2, n2
    if best_n <= 1e-8:
        return best_x

    # Phase 3: Nelder-Mead from Newton result
    if _time_ok():
        x3, n3 = _nelder_mead(F, x1, max_iter=200)
        if n3 < best_n:
            best_x, best_n = x3, n3
        if best_n <= 1e-8:
            return best_x

    # Phase 4: Newton polish from best
    if _time_ok():
        x4, n4 = _newton_broyden(F, best_x, max_iter=30)
        if n4 < best_n:
            best_x, best_n = x4, n4

    return best_x
# EVOLVE-BLOCK-END
