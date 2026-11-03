"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    # Hermite polynomial values at 0
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 9240.0

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    # Precompute grid
    x = np.linspace(-18, 18, 10000)
    dx = x[1] - x[0]

    # Precompute Hermite polynomials on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)

    # Precompute products
    H0g = H0 * gauss
    H4g = H4 * gauss
    H8g = H8 * gauss
    H12g = H12 * gauss

    def compute_bound(c0, c4, c8):
        if time.time() > deadline:
            return 1e10
        c12 = get_c12(c0, c4, c8)

        # g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        g = c0 * H0g + c4 * H4g + c8 * H8g + c12 * H12g

        # L2 norm squared
        norm_sq = np.sum(g**2) * dx
        if norm_sq < 1e-15:
            return 1e10

        # Normalize to L2 = 1
        g_norm = g / np.sqrt(norm_sq)

        # L1 norm of g
        l1_g = np.sum(np.abs(g_norm)) * dx

        # For even functions with n ≡ 0 mod 4, Fourier transform has same form
        # ||ĝ||₁ = ||g||₁ (by symmetry)
        # Uncertainty product bound ~ l1_g * l1_g = l1_g^2
        return l1_g * l1_g

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    best_score = compute_bound(c0_init, c4_init, c8_init)
    best_coeffs = [c0_init, c4_init, c8_init]

    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60.0:
        try:
            bounds = [
                (c0_init - 2.0, c0_init + 2.0),
                (c4_init - 0.5, c4_init + 0.5),
                (c8_init - 0.05, c8_init + 0.05)
            ]
            
            de_result = differential_evolution(
                lambda c: compute_bound(*c),
                bounds=bounds,
                seed=42,
                maxiter=150,
                tol=1e-10,
                popsize=25,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=True,
                init='latinhypercube'
            )
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x.tolist()
                
        except Exception:
            pass

    # Phase 2: Multi-start local optimization around best found
    np.random.seed(42)
    n_starts = 80
    max_iter_per_start = 500
    
    for i in range(n_starts):
        if time.time() > deadline - 20.0:
            break

        # Generate initial guesses around the best found solution
        if i == 0:
            x0 = best_coeffs[:]
        else:
            # Explore with decreasing scales
            scale = max(0.005, 1.0 * (1.0 / (1 + i * 0.05)))
            x0 = [
                best_coeffs[0] + np.random.randn() * scale * 0.5,
                best_coeffs[1] + np.random.randn() * scale * 0.1,
                best_coeffs[2] + np.random.randn() * scale * 0.01
            ]

        try:
            res = minimize(
                lambda c: compute_bound(*c),
                x0,
                method='Nelder-Mead',
                options={'maxiter': max_iter_per_start, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )

            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()

        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0 and best_score < 1e10:
        try:
            res = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.tolist()
        except Exception:
            pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
