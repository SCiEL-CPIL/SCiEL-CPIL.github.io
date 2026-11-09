"""Root finder for 3 fixed 2D systems; hybrid Newton + Nelder-Mead with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _fd_jacobian(F, x, h=1e-7):
    J = np.empty((2, 2), dtype=float)
    Fx = F(x)
    for j in range(2):
        e = np.zeros(2, dtype=float)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h
    return J


def _newton(F, x, max_iter=50, tol=1e-8, deadline=None):
    x = np.array(x, dtype=float)
    best_x = x.copy()
    best_norm = float('inf')
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            break
        nrm = float(np.linalg.norm(Fx))
        if nrm < best_norm:
            best_norm = nrm
            best_x = x.copy()
        if nrm <= tol:
            return x, True
        try:
            J = _fd_jacobian(F, x)
        except Exception:
            break
        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        # backtracking line search on ||F||
        t = 1.0
        Fx_base = Fx
        for _ in range(30):
            x_new = x - t * step
            if not np.all(np.isfinite(x_new)):
                t *= 0.5
                continue
            Fx_new = F(x_new)
            if not np.all(np.isfinite(Fx_new)):
                t *= 0.5
                continue
            nrm_new = float(np.linalg.norm(Fx_new))
            if nrm_new < best_norm:
                best_norm = nrm_new
                best_x = x_new.copy()
            if nrm_new < nrm:
                x = x_new
                break
            t *= 0.5
        else:
            break
    return best_x, (best_norm <= tol)


def _nelder_mead(F, x, deadline):
    def obj(x):
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return 1e300
        return float(np.dot(Fx, Fx))

    res = None
    try:
        res = minimize(obj, np.array(x, dtype=float), method='Nelder-Mead',
                       options={'xatol': 1e-12, 'fatol': 1e-16, 'maxiter': 2000,
                                'maxfev': 4000, 'adaptive': True})
    except Exception:
        pass
    if res is not None and hasattr(res, 'x') and np.all(np.isfinite(res.x)):
        return np.asarray(res.x, dtype=float)
    return np.array(x, dtype=float)


def find_root_system(F, x0):
    deadline = time.time() + 110.0
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2, dtype=float)

    # Phase 1: Newton with backtracking
    x, ok = _newton(F, x, max_iter=50, tol=1e-8, deadline=deadline)
    if ok:
        return x

    # Phase 2: derivative-free Nelder-Mead on ||F||^2
    if time.time() < deadline - 5.0:
        x2 = _nelder_mead(F, x, deadline)
        # Phase 3: Newton polish from Nelder-Mead result
        x3, ok3 = _newton(F, x2, max_iter=50, tol=1e-8, deadline=deadline)
        if ok3:
            return x3
        # pick better of x2 and x3 by residual
        try:
            n2 = float(np.linalg.norm(F(x2)))
            n3 = float(np.linalg.norm(F(x3)))
            if n3 < n2:
                return x3
            return x2
        except Exception:
            return x2

    return x
# EVOLVE-BLOCK-END
