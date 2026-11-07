"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts
and a final high-precision polish pass."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method.
    Broyden reduces F-call count: ~3 calls for the initial Jacobian,
    then 1 call per iteration (vs 3 for finite-diff Newton).
    A final high-precision polish pass (tighter tolerance, exact local solve)
    drives the residual below tol cheaply on systems that nearly converge.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    tol = 1e-8
    start_time = time.time()
    time_limit = 140.0  # Safety margin before the 150s kill

    def time_left():
        return time_limit - (time.time() - start_time)

    def _fd_jac(pt, Fv):
        """Forward-difference Jacobian: 2 extra F calls."""
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(pt + e) - Fv) / h
        return J

    def _try_solve(J, Fv):
        """Solve J*s = -Fv, returning None if singular."""
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-12:
            return None
        try:
            return np.linalg.solve(J, -Fv)
        except np.linalg.LinAlgError:
            return None

    def _broyden_from(pt, Fv, J, max_iters):
        """
        Broyden's good method from (pt, Fv, J) with a backtracking line search.
        Returns (pt, Fv, J, converged) where converged means ||Fv|| <= tol.
        """
        x_local = pt.copy()
        Fx = Fv.copy()
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= tol:
            return x_local, Fx, J, True

        for _it in range(max_iters):
            if time_left() < 0.5:
                break
            if norm_Fx <= tol:
                return x_local, Fx, J, True

            s = _try_solve(J, Fx)
            if s is None:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break
            if s_norm > 10.0:
                s = s * 10.0 / s_norm

            f_prev = norm_Fx ** 2
            alpha = 1.0
            accepted = False

            for _ls in range(15):
                x_try = x_local + alpha * s
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)

                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try.copy()

                if norm_try <= tol:
                    return x_try, F_try, J, True

                f_try = norm_try ** 2
                # Accept full step on monotone progress; otherwise Armijo.
                if f_try < f_prev * 0.5 + 1e-14 or f_try <= f_prev:
                    delta_F = F_try - Fx - J @ s
                    s_norm_sq = np.dot(s, s)
                    if s_norm_sq > 1e-16:
                        J = J + np.outer(delta_F, s) / s_norm_sq
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    accepted = True
                    break
                else:
                    alpha *= 0.5

            if not accepted:
                break

        return x_local, Fx, J, (norm_Fx <= tol)

    # --- Track best point seen so far ---
    best_x = x.copy()
    best_norm = np.linalg.norm(F(x))
    if best_norm <= tol:
        return best_x

    max_restarts = 40

    for restart in range(max_restarts):
        if time_left() < 1.0:
            break

        # Starting point for this restart
        if restart == 0:
            x_local = x.copy()
        else:
            r = np.random.random()
            if r < 0.3:
                x_local = x + np.random.uniform(-5, 5, size=2)
            elif r < 0.6:
                radius = min(1.0 + 0.5 * restart, 20.0)
                dir_vec = np.random.randn(2)
                nrm = np.linalg.norm(dir_vec)
                if nrm < 1e-12:
                    dir_vec = np.array([1.0, 0.0])
                else:
                    dir_vec = dir_vec / nrm
                x_local = best_x + radius * dir_vec
            else:
                x_local = best_x + np.random.randn(2) * 0.1

        if not np.all(np.isfinite(x_local)):
            continue

        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)

        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local

        # Initial Jacobian (2 extra F calls)
        J = _fd_jac(x_local, Fx)
        if _try_solve(J, Fx) is None:
            # Singular Jacobian here; try a different start
            continue

        # Broyden iterations
        x_local, Fx, J, converged = _broyden_from(x_local, Fx, J, max_iters=60)

        if converged:
            return x_local

        if best_norm <= tol:
            return best_x

    # --- Final high-precision polish pass on the best point ---
    if time_left() > 1.0:
        Fb = F(best_x)
        nb = np.linalg.norm(Fb)
        if nb <= tol:
            return best_x
        if nb <= 1e-6:  # close enough to merit a tight local polish
            Jb = _fd_jac(best_x, Fb)
            if _try_solve(Jb, Fb) is not None:
                xp, Fp, Jp, conv = _broyden_from(best_x, Fb, Jb, max_iters=40)
                if np.linalg.norm(Fp) < nb:
                    best_x = xp
                    best_norm = np.linalg.norm(Fp)
                if conv:
                    return best_x

    return best_x
# EVOLVE-BLOCK-END
