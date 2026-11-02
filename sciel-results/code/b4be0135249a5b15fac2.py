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
from scipy.optimize import minimize, basin_hopping


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
    x = np.linspace(-10, 10, 8001)
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

    def compute_l1(coeffs):
        """Compute L1 norm of g(x) = P(x) * exp(-x^2/2)."""
        a0, a4, a8 = coeffs
        a12 = (a0 - H4_0 * a4 + H8_0 * a8) / H12_0
        P_vals = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        g = P_vals * gauss
        return np.sum(np.abs(g)) * dx

    def compute_min_abs_hat(coeffs):
        """Compute minimum |hat_g(k)| where hat_g is Fourier transform of g.
        For this class of functions, hat_g(k) = Q(k) * exp(-k^2/2) where
        Q uses -a12 instead of +a12 due to (-i)^12 = -1."""
        a0, a4, a8 = coeffs
        a12 = (a0 - H4_0 * a4 + H8_0 * a8) / H12_0
        Q_vals = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12
        q = Q_vals * gauss
        return np.min(np.abs(q))

    def objective(coeffs):
        """
        Compute the uncertainty bound: A(f)*A(Fhat(f)) ~ L1(g) / min|hat_g|
        Lower is better.
        """
        l1_val = compute_l1(coeffs)
        min_hat = compute_min_abs_hat(coeffs)
        
        if min_hat < 1e-15 or l1_val < 1e-15:
            return 1e10
        
        return l1_val / min_hat

    def objective_soft_min(coeffs, beta=100.0):
        """
        Smooth surrogate using soft-min for min|hat_g|.
        """
        a0, a4, a8 = coeffs
        a12 = (a0 - H4_0 * a4 + H8_0 * a8) / H12_0
        
        P_vals = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        Q_vals = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12
        
        g = P_vals * gauss
        q = Q_vals * gauss
        
        l1_val = np.sum(np.abs(g)) * dx
        
        if l1_val < 1e-15:
            return 1e10
        
        abs_q = np.abs(q)
        max_abs_q = np.max(abs_q)
        if max_abs_q < 1e-15:
            return 1e10
        
        # Soft min: -1/beta * log(sum(exp(-beta * abs_q)))
        scaled = -beta * (abs_q - max_abs_q)
        scaled = np.clip(scaled, -100, 100)
        log_sum = np.log(np.sum(np.exp(scaled)))
        soft_min_val = -(log_sum + beta * max_abs_q) / beta
        
        if soft_min_val < 1e-15:
            return 1e10
        
        return l1_val / soft_min_val

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
    for _ in range(20):
        point = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-2, 2),
            np.random.uniform(-1, 1)
        ])
        initial_points.append(point)

    # Try each initial point with multiple local methods
    methods = ['Nelder-Mead', 'Powell']

    for point in initial_points:
        if time.time() > deadline - 15.0:
            break

        for method in methods:
            if time.time() > deadline - 15.0:
                break

            try:
                if method == 'Nelder-Mead':
                    result = minimize(
                        objective_soft_min,
                        point,
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    val = objective(result.x)
                    if val < best_val:
                        best_val = val
                        best_coeffs = result.x.copy()
                elif method == 'Powell':
                    result = minimize(
                        objective_soft_min,
                        point,
                        method='Powell',
                        options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12}
                    )
                    val = objective(result.x)
                    if val < best_val:
                        best_val = val
                        best_coeffs = result.x.copy()

            except Exception:
                pass

    # Basin Hopping for global exploration
    if time.time() < deadline - 25.0:
        try:
            def bh_objective(x):
                return objective(x)

            def bh_step(x, T):
                step_size = 0.2 * np.random.randn(3)
                new_x = x + step_size
                return new_x

            bh_result = basin_hopping(
                bh_objective,
                best_coeffs.copy(),
                step=bh_step,
                niter=80,
                T=0.3,
                miniter=50,
                maxiter=500,
                seed=42
            )
            if bh_result.fun < best_val:
                best_val = bh_result.fun
                best_coeffs = bh_result.x.copy()
        except Exception:
            pass

    # Final refinement with very tight tolerances using exact objective
    if time.time() < deadline - 5.0:
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
