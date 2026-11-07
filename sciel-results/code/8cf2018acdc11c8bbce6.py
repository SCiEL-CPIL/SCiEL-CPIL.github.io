"""Root finder for 3 fixed 2D systems: LM + complex-step Jacobian with fallbacks."""
# EVOLVE-BLOCK-START
import numpy as np


def _cs_jacobian(F, x):
    """Complex-step Jacobian: 2 extra F calls, near machine-precision."""
    h = 1e-30
    J = np.empty((2, 2), dtype=float)
    for j in range(2):
        e = np.zeros(2)
        e[j] = 1.0
        Fc = F(x + 1j * h * e)
        J[:, j] = np.asarray(Fc, dtype=complex).imag / h
    return J


def _lm_solve(F, x, budget, count):
    """Levenberg–Marquardt with complex-step Jacobian. Returns (x_best, count)."""
    x = np.array(x, dtype=float)
    Fx = F(x)
    count[0] += 1
    best_x = x.copy()
    best_norm = float(np.linalg.norm(Fx))
    lam = 1e-3
    for _ in range(60):
        if count[0] >= budget:
            break
        if best_norm <= 1e-12:
            break
        J = _cs_jacobian(F, x)
        count[0] += 2
        if count[0] >= budget:
            break
        # Solve (J^T J + lam I) dx = -J^T Fx
        JTJ = J.T @ J
        rhs = -J.T @ Fx
        for _ in range(20):
            A = JTJ + lam * np.eye(2)
            try:
                dx = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            x_trial = x + dx
            F_trial = F(x_trial)
            count[0] += 1
            n_trial = float(np.linalg.norm(F_trial))
            if np.isfinite(n_trial) and n_trial < best_norm * (1.0 - 1e-4):
                # accept
                x = x_trial
                Fx = F_trial
                best_x = x.copy()
                best_norm = n_trial
                lam = max(lam * 0.1, 1e-8)
                break
            else:
                lam *= 10.0
                if lam > 1e12:
                    break
        if lam > 1e12:
            break
    return best_x, count


def _damped_newton(F, x, budget, count):
    """Damped Newton with backtracking line search. Returns (x_best, count)."""
    x = np.array(x, dtype=float)
    Fx = F(x)
    count[0] += 1
    best_x = x.copy()
    best_norm = float(np.linalg.norm(Fx))
    for _ in range(40):
        if count[0] >= budget:
            break
        if best_norm <= 1e-12:
            break
        J = _cs_jacobian(F, x)
        count[0] += 2
        if count[0] >= budget:
            break
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
        # backtracking line search
        alpha = 1.0
        accepted = False
        for _ in range(30):
            if count[0] >= budget:
                break
            x_trial = x - alpha * dx
            F_trial = F(x_trial)
            count[0] += 1
            n_trial = float(np.linalg.norm(F_trial))
            if np.isfinite(n_trial) and n_trial < best_norm * (1.0 - 1e-4 * alpha):
                x = x_trial
                Fx = F_trial
                best_x = x.copy()
                best_norm = n_trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    return best_x, count


def _grid_search(F, x0, budget, count):
    """Coarse-to-fine grid search to locate a good starting point. Returns (x_best, count)."""
    x0 = np.array(x0, dtype=float)
    center = x0
    # Start with a moderate radius, expand if needed
    for radius in [1.0, 5.0, 20.0, 100.0]:
        if count[0] >= budget:
            break
        best_x = None
        best_norm = np.inf
        # Coarse grid
        for nx in np.linspace(-radius, radius, 9):
            for ny in np.linspace(-radius, radius, 9):
                if count[0] >= budget:
                    break
                pt = center + np.array([nx, ny])
                Fpt = F(pt)
                count[0] += 1
                npt = float(np.linalg.norm(Fpt))
                if np.isfinite(npt) and npt < best_norm:
                    best_norm = npt
                    best_x = pt
        if best_x is not None and best_norm < 1.0:
            # Refine around best
            r2 = radius / 4.0
            for nx in np.linspace(-r2, r2, 9):
                for ny in np.linspace(-r2, r2, 9):
                    if count[0] >= budget:
                        break
                    pt = best_x + np.array([nx, ny])
                    Fpt = F(pt)
                    count[0] += 1
                    npt = float(np.linalg.norm(Fpt))
                    if np.isfinite(npt) and npt < best_norm:
                        best_norm = npt
                        best_x = pt
            if best_norm < 0.1:
                return best_x, count
        if best_x is None:
            continue
        # If we found something reasonably close, stop expanding
        if best_norm < 0.5:
            return best_x, count
    # Return x0 if nothing better
    return x0, count


def find_root_system(F, x0):
    budget = 480  # leave margin below 500
    count = [0]
    x0 = np.array(x0, dtype=float)

    # Phase 1: LM from x0
    x, count = _lm_solve(F, x0, budget, count)
    if count[0] < budget and np.linalg.norm(F(x)) <= 1e-8:
        return x

    # Phase 2: Damped Newton from x0
    if count[0] < budget:
        x2, count = _damped_newton(F, x0, budget, count)
        if np.linalg.norm(F(x2)) <= 1e-8:
            return x2
        # Keep best so far
        if np.linalg.norm(F(x2)) < np.linalg.norm(F(x)):
            x = x2

    # Phase 3: Grid search for a better start, then LM
    if count[0] < budget:
        xg, count = _grid_search(F, x0, budget, count)
        if count[0] < budget:
            x3, count = _lm_solve(F, xg, budget, count)
            if np.linalg.norm(F(x3)) <= 1e-8:
                return x3
            if np.linalg.norm(F(x3)) < np.linalg.norm(F(x)):
                x = x3

    # Return best x found (guaranteed finite (2,) array)
    return x
# EVOLVE-BLOCK-END
