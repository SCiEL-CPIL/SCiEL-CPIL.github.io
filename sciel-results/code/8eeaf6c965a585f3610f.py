"""
Combined approach: use the multi-start + multi-method local optimization
framework from the current program, but replace the Gauss-Hermite quadrature
with a dense uniform grid (inspiration 1's approach) which appears to give
better scores (0.152 vs 0.145). Also incorporate the probabilists' Hermite
polynomials from inspiration 1, which may be more numerically stable for
this specific problem.

Key structural change: switch from hermgauss quadrature to a dense uniform
grid with explicit Hermite polynomial evaluation, and use a broader, more
structured set of initial points for the multi-start optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 210s wall clock

    # Seed coefficients from the original program
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # --- Mathematical Setup ---
    # g(x) = P(x) * exp(-x^2/2)
    # P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # Constraint: P(0) = 0 => a0 - 3*a4 + 105*a8 - 10395*a12 = 0
    # a12 = (a0 - 3*a4 + 105*a8) / 10395

    # Fourier domain: Q(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) - a12*H12(x)
    # because (-i)^12 = -1

    # --- Numerical Integration Setup ---
    # Use a dense uniform grid (inspiration 1 approach)
    x = np.linspace(-8, 8, 4001)
    dx = x[1] - x[0]
    gauss = np.exp(-x**2 / 2)

    # Probabilists' Hermite polynomials (inspiration 1)
    def he0(x):
        return np.ones_like(x)

    def he4(x):
        return x**4 - 6*x**2 + 3

    def he8(x):
        return x**8 - 28*x**6 + 420*x**4 - 840*x**2 + 105

    def he12(x):
        return x**12 - 66*x**10 + 2310*x**8 - 39900*x**6 + 332640*x**4 - 798336*x**2 + 10395

    H0 = he0(x)
    H4 = he4(x)
    H8 = he8(x)
    H12 = he12(x)

    # Values at 0 for constraint
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0

    def objective(coeffs):
        """
        Compute the uncertainty bound for given coefficients a0, a4, a8.
        Uses the L4/L2 norm ratio approach.
        """
        a0, a4, a8 = coeffs

        # Calculate a12 to satisfy P(0) = 0
        a12 = (a0 - H4_0 * a4 + H8_0 * a8) / H12_0

        # Construct P(x) and Q(x)
        P_vals = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        Q_vals = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12

        g = P_vals * gauss
        q = Q_vals * gauss

        # Compute norms
        norm2_g_sq = np.sum(g**2) * dx
        norm2_q_sq = np.sum(q**2) * dx
        norm4_g_4 = np.sum(g**4) * dx
        norm4_q_4 = np.sum(q**4) * dx

        # Avoid division by zero or invalid values
        if (norm2_g_sq < 1e-12 or norm2_q_sq < 1e-12 or
                norm4_g_4 < 1e-12 or norm4_q_4 < 1e-12):
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = np.sqrt(np.sqrt(norm4_g_4))
        norm4_q = np.sqrt(np.sqrt(norm4_q_4))

        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        return bound

    # --- Optimization Strategy: Multi-start with diverse initial points ---

    best_coeffs = seed_coeffs.copy()
    best_val = objective(seed_coeffs)

    # Generate diverse initial points
    np.random.seed(12345)
    initial_points = [seed_coeffs.copy()]

    # Add points around the seed with varying scales
    for scale in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        for _ in range(3):
            perturbation = np.random.randn(3) * scale
            point = seed_coeffs + perturbation
            initial_points.append(point)

    # Add some points in a broader range
    for _ in range(15):
        point = np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-1, 1),
            np.random.uniform(-0.5, 0.5)
        ])
        initial_points.append(point)

    # Try each initial point with multiple local methods
    methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']

    for point in initial_points:
        if time.time() > deadline - 5.0:
            break

        for method in methods:
            if time.time() > deadline - 5.0:
                break

            try:
                if method == 'Nelder-Mead':
                    result = minimize(
                        objective,
                        point,
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                elif method == 'Powell':
                    result = minimize(
                        objective,
                        point,
                        method='Powell',
                        options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12}
                    )
                else:  # L-BFGS-B
                    result = minimize(
                        objective,
                        point,
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-8}
                    )

                if result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = result.x.copy()

            except Exception:
                pass

    # Final refinement with very tight tolerances
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
