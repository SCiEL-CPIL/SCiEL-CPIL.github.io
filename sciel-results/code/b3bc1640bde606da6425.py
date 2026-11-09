"""Root finder for 3 fixed two-dimensional systems using a robust hybrid approach."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _safe_F(F, x):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _broyden_update(J, x_new, x_old, F_new, F_old):
    """Rank-1 Broyden update."""
    s = x_new - x_old
    y = F_new - F_old
    denom = np.dot(y, s)
    if abs(denom) > 1e-14:
        J = J + np.outer(y - J @ s, s) / denom
    return J


def _newton_polish(F, x, tol=1e-12, max_iter=20, deadline=None):
    """Newton's method with finite-difference Jacobian for high-precision polish."""
    x = np.array(x, dtype=float)
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        Fx = _safe_F(F, x)
        if Fx is None:
            break
        if np.linalg.norm(Fx) <= tol:
            return x
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-7
            Fx_e = _safe_F(F, x + e)
            if Fx_e is None:
                break
            J[:, j] = (Fx_e - Fx) / 1e-7
        else:
            try:
                x = x - np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(x)):
                break
    return x


def _nelder_mead(F, x0, max_iter=200, deadline=None):
    """Nelder-Mead on Phi(x) = 0.5*||F(x)||^2 for robustness."""
    def objective(x):
        Fx = _safe_F(F, x)
        if Fx is None:
            return 1e10
        return 0.5 * np.dot(Fx, Fx)

    x0 = np.array(x0, dtype=float)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(2):
        dx = x0.copy()
        dx[i] += 0.1 * max(1.0, abs(x0[i]))
        simplex.append(dx)

    fvals = [objective(v) for v in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[0] < 1e-24:
            break

        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection
        xr = centroid + alpha * (centroid - worst)
        fr = objective(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
            continue

        # Expansion
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - worst)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
            continue

        # Contraction
        if fr < fvals[-1]:
            xc = centroid + rho * (worst - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
            continue
        else:
            # Shrink
            for i in range(1, len(simplex)):
                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                fvals[i] = objective(simplex[i])

    order = np.argsort(fvals)
    return simplex[order[0]]


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf

    # Phase 1: Broyden's method
    F0 = _safe_F(F, x)
    if F0 is None:
        # Try Nelder-Mead fallback
        x_nm = _nelder_mead(F, x0, max_iter=100, deadline=deadline)
        F_nm = _safe_F(F, x_nm)
        if F_nm is not None and np.linalg.norm(F_nm) < best_norm:
            best_x = x_nm
            best_norm = np.linalg.norm(F_nm)
        return best_x

    best_norm = np.linalg.norm(F0)
    if best_norm <= 1e-8:
        return x

    # Initialize Jacobian with finite differences
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = 1e-6
        Fj = _safe_F(F, x + e)
        if Fj is None:
            break
        J[:, j] = (Fj - F0) / 1e-6
    else:
        # Broyden iterations
        for _ in range(50):
            if time.time() > deadline:
                break
            try:
                dx = np.linalg.solve(J, F0)
            except np.linalg.LinAlgError:
                break
            x_new = x - dx
            if not np.all(np.isfinite(x_new)):
                break
            F_new = _safe_F(F, x_new)
            if F_new is None:
                break
            norm_new = np.linalg.norm(F_new)
            if norm_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_new
            if norm_new <= 1e-10:
                x = x_new
                F0 = F_new
                break
            J = _broyden_update(J, x_new, x, F_new, F0)
            x = x_new
            F0 = F_new
            if np.linalg.norm(x - x0) > 1e6:
                break

    # Phase 2: Newton polish if close
    if best_norm <= 1e-6 and time.time() < deadline:
        x_polished = _newton_polish(F, best_x, tol=1e-12, max_iter=20, deadline=deadline)
        F_polished = _safe_F(F, x_polished)
        if F_polished is not None:
            norm_polished = np.linalg.norm(F_polished)
            if norm_polished < best_norm:
                best_x = x_polished
                best_norm = norm_polished

    # Phase 3: If still not close, try Nelder-Mead
    if best_norm > 1e-8 and time.time() < deadline:
        x_nm = _nelder_mead(F, best_x, max_iter=150, deadline=deadline)
        F_nm = _safe_F(F, x_nm)
        if F_nm is not None:
            norm_nm = np.linalg.norm(F_nm)
            if norm_nm < best_norm:
                best_x = x_nm
                best_norm = norm_nm
                # Polish the Nelder-Mead result
                if best_norm <= 1e-6 and time.time() < deadline:
                    x_polished = _newton_polish(F, best_x, tol=1e-12, max_iter=20, deadline=deadline)
                    F_polished = _safe_F(F, x_polished)
                    if F_polished is not None:
                        norm_polished = np.linalg.norm(F_polished)
                        if norm_polished < best_norm:
                            best_x = x_polished
                            best_norm = norm_polished

    return best_x
# EVOLVE-BLOCK-END
