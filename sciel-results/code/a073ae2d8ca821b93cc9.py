"""
Rewritten program: multi-start local optimization with diverse initial points
and exploitation of problem structure for uncertainty bound minimization.

Strategy:
1. Use Gauss-Hermite quadrature for exact integration.
2. Generate diverse starting points using stratified sampling in a bounded region.
3. Run multiple rounds of local optimization (Nelder-Mead) from each starting point.
4. Keep track of the best solution found and return it.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Leave 15s margin for safety

    # Seed from the problem description
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # --- Mathematical Setup ---
    # g(x) = P(x) * exp(-x^2/2)
    # P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # Constraint: P(0) = 0 => a12 = (a0 - 3*a4 + 105*a8) / 10395
    # Fourier domain: Q(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) - a12*H12(x)
    # Objective: (||P e^{-x^2/2}||_4 * ||Q e^{-x^2/2}||_4) / (||P e^{-x^2/2}||_2 * ||Q e^{-x^2/2}||_2)

    # --- Numerical Integration Setup ---
    # Use Gauss-Hermite quadrature with weight e^{-x^2}
    # Transform: x = sqrt(2)*t, so e^{-x^2/2} = e^{-t^2}, dx = sqrt(2) dt
    # Integral = sqrt(2) * sum(w_i * f(sqrt(2)*t_i))
    
    N_QUAD = 32  # Use 32 nodes for precision (degree 48 polynomials need N>=25)
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

    # --- Multi-start Local Optimization ---
    
    best_coeffs = seed_coeffs.copy()
    best_val = objective(seed_coeffs)

    # Define search bounds
    bounds = [(-3.0, 3.0), (-2.0, 2.0), (-0.5, 0.5)]

    # Generate diverse starting points using stratified sampling
    rng = np.random.default_rng(42)
    n_starts = 20
    start_points = []
    
    # Include the seed point
    start_points.append(seed_coeffs.copy())
    
    # Add stratified random points within bounds
    for i in range(n_starts - 1):
        point = np.array([
            bounds[0][0] + (bounds[0][1] - bounds[0][0]) * (i + 1) / (n_starts - 1),
            bounds[1][0] + (bounds[1][1] - bounds[1][0]) * rng.random(),
            bounds[2][0] + (bounds[2][1] - bounds[2][0]) * rng.random()
        ])
        # Add some jitter for diversity
        point += rng.normal(0, 0.1, size=3)
        # Clip to bounds
        for j in range(3):
            point[j] = np.clip(point[j], bounds[j][0], bounds[j][1])
        start_points.append(point)
    
    # Add some additional diverse points
    extra_starts = [
        np.array([0.0, 0.1, 0.01]),
        np.array([-0.5, 0.2, 0.05]),
        np.array([0.5, -0.1, 0.02]),
        np.array([-1.0, 0.3, 0.03]),
        np.array([1.0, -0.2, 0.04]),
    ]
    start_points.extend(extra_starts)

    # Run local optimization from each starting point
    for start_idx, start_point in enumerate(start_points):
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(
                objective,
                start_point,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
                
        except Exception:
            pass

    # Final refinement with Powell method from the best point found
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
