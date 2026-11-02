"""
Structural rewrite: Basin hopping with adaptive local refinement for the
Hermite-Gaussian uncertainty bound minimization.

Key structural changes from DE approach:
1. Exploit scale invariance: normalize coefficients so a0=1 (or handle sign)
2. Use basin_hopping which combines L-BFGS-B local optimization with global
   perturbations in a systematic way
3. Multiple independent basin hopping runs with different perturbation schedules
4. Direct mathematical structure: the objective is a ratio of L4/L2 norms
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, basin_hopping
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 200.0  # Leave 10s margin for safety

    # --- Mathematical Setup ---
    # g(x) = P(x) * exp(-x^2/2)
    # P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # Constraint: P(0) = 0 => a12 = (a0 - 3*a4 + 105*a8) / 10395
    # Fourier domain: Q(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) - a12*H12(x)
    # Objective: (||P e^{-x^2/2}||_4 * ||Q e^{-x^2/2}||_4) / (||P e^{-x^2/2}||_2 * ||Q e^{-x^2/2}||_2)

    # --- Numerical Integration Setup ---
    N_QUAD = 50  # Use 50 nodes for high precision
    nodes, weights = hermgauss(N_QUAD)
    x_nodes = np.sqrt(2.0) * nodes
    w_scaled = np.sqrt(2.0) * weights

    def hermite_physicist(x, n):
        """Compute Physicists' Hermite polynomial H_n(x) using recurrence."""
        if n == 0:
            return np.ones_like(x, dtype=float)
        elif n == 1:
            return 2.0 * x
        else:
            H_nm1 = np.ones_like(x, dtype=float)
            H_n = 2.0 * x
            for k in range(2, n + 1):
                H_np1 = 2.0 * x * H_n - 2.0 * (k - 1) * H_nm1
                H_nm1 = H_n
                H_n = H_np1
            return H_n

    # Precompute Hermite polynomials at quadrature nodes
    H0_nodes = hermite_physicist(x_nodes, 0)
    H4_nodes = hermite_physicist(x_nodes, 4)
    H8_nodes = hermite_physicist(x_nodes, 8)
    H12_nodes = hermite_physicist(x_nodes, 12)

    def objective(coeffs):
        """Compute the uncertainty bound for given coefficients a0, a4, a8."""
        a0, a4, a8 = coeffs
        
        # Calculate a12 to satisfy P(0) = 0
        a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

        # Construct P(x) and Q(x) at nodes
        P_vals = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes + a12 * H12_nodes
        Q_vals = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes - a12 * H12_nodes

        # Compute squared L2 norms and fourth-power L4 norms
        norm2_g_sq = np.sum(w_scaled * P_vals**2)
        norm2_q_sq = np.sum(w_scaled * Q_vals**2)
        norm4_g_4 = np.sum(w_scaled * P_vals**4)
        norm4_q_4 = np.sum(w_scaled * Q_vals**4)

        # Avoid division by zero and invalid values
        if norm2_g_sq < 1e-12 or norm2_q_sq < 1e-12 or norm4_g_4 < 1e-12 or norm4_q_4 < 1e-12:
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = norm4_g_4 ** 0.25
        norm4_q = norm4_q_4 ** 0.25

        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        return bound

    # --- Seed point ---
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = seed_coeffs.copy()
    best_val = objective(seed_coeffs)

    # --- Strategy: Multiple basin hopping runs with different schedules ---
    # The objective is scale-invariant (homogeneous of degree 0), so we can
    # normalize. But we'll work in the original space with careful bounds.

    # Define a local optimizer that uses L-BFGS-B
    def local_optimizer(x):
        result = minimize(
            objective, x, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        return result.fun, result.x, {}

    # Run multiple basin hopping instances with different parameters
    bh_configs = [
        {'step': 0.5, 'niter': 50, 'T': 0.1},
        {'step': 0.3, 'niter': 80, 'T': 0.05},
        {'step': 0.8, 'niter': 40, 'T': 0.2},
        {'step': 0.2, 'niter': 100, 'T': 0.02},
        {'step': 1.0, 'niter': 30, 'T': 0.3},
        {'step': 0.4, 'niter': 60, 'T': 0.08},
    ]

    # Also try starting from different initial points
    initial_points = [
        seed_coeffs,
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-0.8, 0.08, 0.002]),
        np.array([-1.5, 0.02, 0.0003]),
        np.array([-0.5, 0.1, 0.003]),
    ]

    # Interleave: for each config, try from the best current point and a random init
    for i, cfg in enumerate(bh_configs):
        if time.time() > deadline - 15.0:
            break

        # Use the current best as starting point for first few, then diverse starts
        if i < 3:
            x0 = best_coeffs.copy()
        else:
            x0 = initial_points[i % len(initial_points)].copy()

        try:
            bh_result = basin_hopping(
                local_optimizer,
                x0,
                stepsize=cfg['step'],
                niter=cfg['niter'],
                T=cfg['T'],
                mininterval=1,
                maxinterval=5,
            )

            if bh_result.fun < best_val:
                best_val = bh_result.fun
                best_coeffs = bh_result.x.copy()

        except Exception:
            pass

    # --- Final local refinement from the best point found ---
    if time.time() < deadline - 3.0:
        try:
            final_result = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if final_result.fun < best_val:
                best_val = final_result.fun
                best_coeffs = final_result.x.copy()
        except Exception:
            pass

    # --- Try Powell from best point as well ---
    if time.time() < deadline - 2.0:
        try:
            powell_result = minimize(
                objective,
                best_coeffs,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-15, 'ftol': 1e-17}
            )
            if powell_result.fun < best_val:
                best_val = powell_result.fun
                best_coeffs = powell_result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
