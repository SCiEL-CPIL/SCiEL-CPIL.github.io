"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s kill

    # Hermite polynomial values at 0
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 9240.0

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    # Precompute grid - use a moderate grid for balance of accuracy and speed
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]
    
    # Precompute Hermite polynomials on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    
    # Precompute weights for trapezoidal rule
    weights = np.ones_like(x)
    weights[0] = 0.5
    weights[-1] = 0.5

    def compute_bound(c0, c4, c8):
        if time.time() > deadline:
            return 1e10
        c12 = get_c12(c0, c4, c8)

        # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        g = poly * gauss

        # L2 norm squared
        norm_sq = np.sum(weights * g**2) * dx
        if norm_sq < 1e-15:
            return 1e10

        # Normalize to L2 = 1
        g_norm = g / np.sqrt(norm_sq)

        # L1 norm of g
        l1_g = np.sum(weights * np.abs(g_norm)) * dx

        # Uncertainty product bound ~ l1_g * l1_g = l1_g^2
        return l1_g * l1_g

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    best_score = compute_bound(c0_init, c4_init, c8_init)
    best_coeffs = [c0_init, c4_init, c8_init]

    # Strategy:
    # 1. Multiple L-BFGS-B starts from diverse initial points
    # 2. Final high-precision polish with tighter tolerances

    # Generate diverse starting points
    starts = [[c0_init, c4_init, c8_init]]
    
    # Add perturbations around seed
    np.random.seed(123)
    for i in range(12):
        if time.time() > deadline - 30.0:
            break
        scale_c0 = 0.3
        scale_c4 = 0.05
        scale_c8 = 0.002
        x0 = [
            c0_init + np.random.randn() * scale_c0,
            c4_init + np.random.randn() * scale_c4,
            c8_init + np.random.randn() * scale_c8
        ]
        starts.append(x0)

    # Add some structured perturbations
    for delta_c0 in [-0.2, 0.2, -0.4, 0.4]:
        if time.time() > deadline - 25.0:
            break
        starts.append([c0_init + delta_c0, c4_init, c8_init])
    
    for delta_c4 in [-0.03, 0.03, -0.06, 0.06]:
        if time.time() > deadline - 25.0:
            break
        starts.append([c0_init, c4_init + delta_c4, c8_init])

    for delta_c8 in [-0.001, 0.001, -0.002, 0.002]:
        if time.time() > deadline - 25.0:
            break
        starts.append([c0_init, c4_init, c8_init + delta_c8])

    # Phase 1: L-BFGS-B from multiple starts
    for x0 in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            res_lbfgs = minimize(
                lambda c: compute_bound(*c),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_lbfgs.fun < best_score:
                best_score = res_lbfgs.fun
                best_coeffs = res_lbfgs.x.tolist()
        except Exception:
            pass

    # Phase 2: Final polish with Nelder-Mead for robust convergence
    if time.time() < deadline - 5.0 and best_score < 1e10:
        try:
            res_nm = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_coeffs = res_nm.x.tolist()
        except Exception:
            pass

    # Phase 3: One more L-BFGS-B pass from the polished point with tighter tolerances
    if time.time() < deadline - 3.0 and best_score < 1e10:
        try:
            res_final = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_coeffs = res_final.x.tolist()
        except Exception:
            pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
