# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Find the real coefficients (c_m) for m = -10..10 of a real degree-10 spherical harmonic
        f = sum_{m=-10}^{10} c_m Y_10^m
    that MINIMIZES the norm ratio ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}.
    Returns a numpy array of shape (21,) with coefficients ordered m = -10, -9, ..., 10.
    """
    k = 10
    start_time = time.time()
    deadline = start_time + 600.0 - 15.0  # 15s safety margin

    # Quadrature setup: 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # Precompute cos/sin of m*phi
    cos_m_phi = np.zeros((11, n_phi))
    sin_m_phi = np.zeros((11, n_phi))
    for m in range(11):
        if m == 0:
            cos_m_phi[m] = 1.0
        else:
            cos_m_phi[m] = np.cos(m * phi)
            sin_m_phi[m] = np.sin(m * phi)

    # Normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * math.factorial(10 - m) / math.factorial(10 + m))

    # Associated Legendre polynomials
    P_legendre = np.zeros((11, 70))
    for m in range(11):
        P_legendre[m] = lpmv(m, 10, x_gl)

    # Build Y_flat matrix of shape (70*130, 21)
    N_q = 70 * n_phi
    Y_flat = np.zeros((N_q, 21), dtype=np.float64)
    w_flat = np.zeros(N_q, dtype=np.float64)

    for i in range(70):
        row_start = i * n_phi
        row_end = (i + 1) * n_phi
        w_flat[row_start:row_end] = w_gl[i] * dphi

        # m = 0
        Y_flat[row_start:row_end, 10] = N[0] * P_legendre[0, i]

        # m > 0 and m < 0
        for m in range(1, 11):
            coeff_m_i = np.sqrt(2.0) * N[m] * P_legendre[m, i]
            Y_flat[row_start:row_end, 10 + m] = coeff_m_i * cos_m_phi[m]
            Y_flat[row_start:row_end, 10 - m] = coeff_m_i * sin_m_phi[m]

    def objective_J(c_vec):
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        if S2 <= 1e-300:
            return 1e100
        return S4 / (S2 * S2)

    def jac_J(c_vec):
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        if S2 <= 1e-300:
            return np.zeros(21)
        wf = w_flat * f
        wf3 = w_flat * f * f2
        dS2 = 2.0 * (Y_flat.T @ wf)
        dS4 = 4.0 * (Y_flat.T @ wf3)
        grad = (S2 * dS4 - 2.0 * S4 * dS2) / (S2 * S2 * S2)
        return grad

    def norm_constraint(c_vec):
        return np.dot(c_vec, c_vec) - 1.0

    def norm_jac(c_vec):
        return 2.0 * c_vec

    cons = [{'type': 'eq', 'fun': norm_constraint, 'jac': norm_jac}]

    best_result = None
    best_obj = np.inf

    # Generate diverse starting points
    starts = []

    # Symmetry-breaking structured starts
    for m_pair in [(0, 10), (0, 9), (0, 8), (10, 9), (10, 8), (10, 7), (9, 8), (8, 7)]:
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            c = np.zeros(21)
            c[10 + m_pair[0]] = ratio
            c[10 + m_pair[1]] = np.sqrt(max(0, 1 - ratio**2))
            starts.append(c)
            c2 = np.zeros(21)
            c2[10 - m_pair[0]] = ratio
            c2[10 - m_pair[1]] = np.sqrt(max(0, 1 - ratio**2))
            starts.append(c2)

    # Pure modes
    for m in range(11):
        c = np.zeros(21)
        c[10 + m] = 1.0
        starts.append(c)
        if m > 0:
            c2 = np.zeros(21)
            c2[10 - m] = 1.0
            starts.append(c2)

    # Random starts with different seeds
    rng = np.random.default_rng(42)
    for _ in range(50):
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        starts.append(c)

    # Optimization loop with time management
    n_starts = len(starts)
    for i, c_start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break

        # Estimate remaining time per remaining start
        remaining_starts = n_starts - i
        time_per_start = (deadline - elapsed) / max(1, remaining_starts)
        maxiter = max(50, int(min(500, time_per_start * 100)))

        try:
            result = minimize(
                objective_J,
                c_start,
                method='SLSQP',
                jac=jac_J,
                constraints=cons,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )

            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x.copy()

        except Exception:
            pass

    if best_result is None:
        best_result = np.ones(21) / np.sqrt(21)

    # Polish: multiple refinement passes from the best result
    elapsed = time.time() - start_time
    if elapsed < deadline - 2.0:
        for polish_iter in range(5):
            if time.time() - start_time > deadline - 2.0:
                break
            try:
                result_polish = minimize(
                    objective_J,
                    best_result,
                    method='SLSQP',
                    jac=jac_J,
                    constraints=cons,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                if result_polish.fun < best_obj:
                    best_obj = result_polish.fun
                    best_result = result_polish.x.copy()
            except Exception:
                break

    # Additional refinement using trust-constr from the best point
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        try:
            from scipy.optimize import LinearConstraint
            bounds = [(-100, 100)] * 21
            lc = LinearConstraint(np.eye(21), 1.0, 1.0)  # This won't work directly for norm constraint
            # Use a different approach: minimize on the sphere using parameterization
            # Skip trust-constr due to complexity; instead do more SLSQP with perturbed starts
            for _ in range(10):
                if time.time() - start_time > deadline - 2.0:
                    break
                c_pert = best_result + 0.01 * np.random.default_rng(123).standard_normal(21)
                c_pert = c_pert / np.linalg.norm(c_pert)
                result = minimize(
                    objective_J,
                    c_pert,
                    method='SLSQP',
                    jac=jac_J,
                    constraints=cons,
                    options={'maxiter': 300, 'ftol': 1e-14}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result.x.copy()
        except Exception:
            pass

    c = best_result

    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4

    return c

# EVOLVE-BLOCK-END
