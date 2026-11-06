"""ODE solver: Convection-diffusion 100-point using BDF with sparse Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.sparse import diags, lil_matrix, csc_matrix


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    deadline = time.time() + 290.0  # safety margin before 330s kill

    # Build a sparse tridiagonal Jacobian for the convection-diffusion operator.
    # We probe f to estimate the structure. For a 100-point convection-diffusion
    # problem, the Jacobian is typically tridiagonal (or close to it).
    # We'll use a finite-difference approach to build the Jacobian once,
    # then pass it to solve_ivp for efficient implicit solves.

    def build_jacobian(t, y):
        """Build sparse Jacobian via finite differences."""
        J = lil_matrix((n, n), dtype=float)
        h = 1e-8 * np.maximum(1.0, np.abs(y))
        y_base = y.copy()
        for j in range(n):
            y_pert = y_base.copy()
            y_pert[j] += h[j]
            f_pert = np.array(f(t, y_pert), dtype=float)
            f_base = np.array(f(t, y_base), dtype=float)
            J[:, j] = (f_pert - f_base) / h[j]
        return J.tocsc()

    # Estimate the Jacobian at the initial point
    try:
        J0 = build_jacobian(t0, y0_arr)
        # Check if it's approximately tridiagonal (common for convection-diffusion)
        nnz = J0.nnz
        # For a tridiagonal 100x100, nnz should be around 298
        # If it's dense, we'll still use it but it won't be as efficient
        use_sparse_jac = nnz < 5 * n  # sparse if fewer than 5n nonzeros
    except Exception:
        use_sparse_jac = False
        J0 = None

    # Primary solve: BDF with sparse Jacobian
    try:
        if use_sparse_jac and J0 is not None:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='BDF',
                rtol=1e-12, atol=1e-14,
                jac=J0,
                max_step=(t1 - t0) / 100.0,
            )
        else:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='BDF',
                rtol=1e-12, atol=1e-14,
                max_step=(t1 - t0) / 100.0,
            )
        if sol.success and sol.y.shape[1] > 0:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                # If we have time, try a polish with tighter tolerance
                if time.time() < deadline - 40.0:
                    try:
                        if use_sparse_jac and J0 is not None:
                            # Rebuild Jacobian at the final point for better accuracy
                            J_final = build_jacobian(t1, y_final)
                            sol2 = solve_ivp(
                                f, (t0, t1), y0_arr,
                                method='BDF',
                                rtol=1e-14, atol=1e-16,
                                jac=J_final,
                                max_step=(t1 - t0) / 100.0,
                            )
                        else:
                            sol2 = solve_ivp(
                                f, (t0, t1), y0_arr,
                                method='BDF',
                                rtol=1e-14, atol=1e-16,
                                max_step=(t1 - t0) / 100.0,
                            )
                        if sol2.success and sol2.y.shape[1] > 0:
                            y_final2 = sol2.y[:, -1]
                            if np.all(np.isfinite(y_final2)):
                                return y_final2
                    except Exception:
                        pass
                return y_final
    except Exception:
        pass

    # Fallback: Radau with sparse Jacobian
    try:
        if use_sparse_jac and J0 is not None:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='Radau',
                rtol=1e-12, atol=1e-14,
                jac=J0,
                max_step=(t1 - t0) / 100.0,
            )
        else:
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='Radau',
                rtol=1e-12, atol=1e-14,
                max_step=(t1 - t0) / 100.0,
            )
        if sol.success and sol.y.shape[1] > 0:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                return y_final
    except Exception:
        pass

    # Fallback: LSODA
    try:
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='LSODA',
            rtol=1e-12, atol=1e-14,
            max_step=(t1 - t0) / 100.0,
        )
        if sol.success and sol.y.shape[1] > 0:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                return y_final
    except Exception:
        pass

    # Last resort: fixed-step RK4
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0_arr.copy()
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    return y
# EVOLVE-BLOCK-END
