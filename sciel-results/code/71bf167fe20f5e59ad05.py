# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    k = 10
    start_time = time.time()
    deadline = start_time + 575.0  # leave 25s safety margin

    # ---- Quadrature setup ----
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    cos_m_phi = np.zeros((11, n_phi))
    sin_m_phi = np.zeros((11, n_phi))
    for m in range(11):
        if m == 0:
            cos_m_phi[m] = 1.0
            sin_m_phi[m] = 0.0
        else:
            cos_m_phi[m] = np.cos(m * phi)
            sin_m_phi[m] = np.sin(m * phi)

    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * math.factorial(10 - m) / math.factorial(10 + m))

    P_legendre = np.zeros((11, 70))
    for m in range(11):
        P_legendre[m] = lpmv(m, 10, x_gl)

    N_q = 70 * n_phi
    Y_flat = np.zeros((N_q, 21), dtype=np.float64)
    w_flat = np.zeros(N_q, dtype=np.float64)

    for i in range(70):
        rs = i * n_phi
        re = (i + 1) * n_phi
        w_flat[rs:re] = w_gl[i] * dphi
        Y_flat[rs:re, 10] = N[0] * P_legendre[0, i]
        for m in range(1, 11):
            coeff = np.sqrt(2.0) * N[m] * P_legendre[m, i]
            Y_flat[rs:re, 10 + m] = coeff * cos_m_phi[m]
            Y_flat[rs:re, 10 - m] = coeff * sin_m_phi[m]

    # ---- Objective (conic/epigraph formulation) ----
    # We minimize S4(c) - t * S2(c)^2 subject to t >= 0.
    # At optimum, t = S4/S2^2 = J(c), recovering the original ratio.
    # This removes the unit-norm equality constraint, making the problem
    # smooth and unconstrained except for t >= 0.

    def S2_of(c):
        f = Y_flat @ c
        return np.sum(w_flat * f * f)

    def S4_of(c):
        f = Y_flat @ c
        f2 = f * f
        return np.sum(w_flat * f2 * f2)

    def objective_conic(x):
        c = x[:21]
        t = x[21]
        f = Y_flat @ c
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        return S4 - t * S2 * S2

    def jac_conic(x):
        c = x[:21]
        t = x[21]
        f = Y_flat @ c
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        # d/dc: dS4/dc - t * 2*S2 * dS2/dc
        wf = w_flat * f
        wf3 = w_flat * f * f2
        dS2 = 2.0 * (Y_flat.T @ wf)
        dS4 = 4.0 * (Y_flat.T @ wf3)
        dc = dS4 - 2.0 * t * S2 * dS2
        dt = -S2 * S2
        return np.concatenate([dc, [dt]])

    # Inequality constraint: t >= 0  =>  -(t) <= 0
    def conic_constraint(x):
        return -x[21]

    def conic_constraint_jac(x):
        g = np.zeros(22)
        g[21] = -1.0
        return g

    cons = [{'type': 'ineq', 'fun': conic_constraint, 'jac': conic_constraint_jac}]

    best_c = None
    best_obj = np.inf

    # ---- Starting points ----
    starts = []

    # Structured starts: single m
    for m_idx in [10, 0, 20, 5, 15]:
        c = np.zeros(21)
        c[m_idx] = 1.0
        starts.append(c)

    # Structured starts: pairs
    for a, b in [(10, 20), (10, 0), (10, 5), (0, 20), (5, 15), (10, 15)]:
        c = np.zeros(21)
        c[a] = 1.0 / np.sqrt(2)
        c[b] = 1.0 / np.sqrt(2)
        starts.append(c)

    # Random starts
    for seed in range(40):
        rng = np.random.RandomState(seed)
        c = rng.randn(21)
        c = c / np.linalg.norm(c)
        starts.append(c)

    # Optimization loop
    for i, c_start in enumerate(starts):
        if time.time() > deadline - 30.0:
            break

        # Initial t estimate
        S2_init = S2_of(c_start)
        S4_init = S4_of(c_start)
        t_init = S4_init / (S2_init * S2_init) if S2_init > 1e-300 else 1.0

        x0 = np.concatenate([c_start, [t_init]])

        try:
            result = minimize(
                objective_conic,
                x0,
                method='SLSQP',
                jac=jac_conic,
                constraints=cons,
                options={'maxiter': 400, 'ftol': 1e-14}
            )

            c_res = result.x[:21]
            # Evaluate actual J
            S2_r = S2_of(c_res)
            S4_r = S4_of(c_res)
            if S2_r > 1e-300:
                J_r = S4_r / (S2_r * S2_r)
            else:
                J_r = np.inf

            if J_r < best_obj:
                best_obj = J_r
                best_c = c_res.copy()

        except Exception:
            pass

    if best_c is None:
        best_c = np.ones(21) / np.sqrt(21.0)

    # ---- Gauss-Newton polish step ----
    # Work in c-space (21D). Minimize J(c) = S4/S2^2.
    # Use Gauss-Newton with trust region and line search.
    if time.time() < deadline - 10.0:
        c = best_c.copy()
        for gn_iter in range(200):
            if time.time() > deadline - 10.0:
                break

            f = Y_flat @ c
            f2 = f * f
            S2 = np.sum(w_flat * f2)
            S4 = np.sum(w_flat * f2 * f2)
            if S2 <= 1e-300:
                break
            J = S4 / (S2 * S2)

            # Gradient of J
            wf = w_flat * f
            wf3 = w_flat * f * f2
            dS2 = 2.0 * (Y_flat.T @ wf)
            dS4 = 4.0 * (Y_flat.T @ wf3)
            grad = (dS4 * S2 - 2.0 * S4 * dS2) / (S2 * S2)

            # Hessian of J (approximate Gauss-Newton)
            # J = S4 * S2^{-2}
            # dJ = (dS4/S2^2) - (2*S4/S2^3)*dS2
            # d2J ≈ (d2S4/S2^2) - (2*S4/S2^3)*d2S2 - (4*S4/S2^4)*(dS2)^2 ... complex
            # Use Gauss-Newton: H ≈ (1/S2^2)*H_S4 - (2*S4/S2^3)*H_S2
            # H_S2 = 4 * Y^T diag(w) Y
            # H_S4 = 16 * Y^T diag(w f^2) Y
            # This is expensive to form explicitly; use the fact that
            # for the Gauss-Newton step we solve H p = -grad
            # Approximate with diagonal scaling + conjugate gradient

            # Simpler: use steepest descent with Armijo line search
            # since full Hessian is expensive
            # But let's try a quasi-Newton via BFGS update

            # Actually, let's just do a careful line search with gradient
            # and adaptive step size
            step_size = 0.01
            for ls_iter in range(50):
                c_trial = c - step_size * grad
                f_t = Y_flat @ c_trial
                f2_t = f_t * f_t
                S2_t = np.sum(w_flat * f2_t)
                S4_t = np.sum(w_flat * f2_t * f2_t)
                if S2_t > 1e-300:
                    J_t = S4_t / (S2_t * S2_t)
                else:
                    J_t = np.inf

                if J_t < J - 1e-4 * step_size * np.dot(grad, grad):
                    c = c_trial
                    break
                step_size *= 0.5

            if step_size < 1e-12:
                break

    c = best_c if best_c is not None else c

    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4

    return c

# EVOLVE-BLOCK-END
