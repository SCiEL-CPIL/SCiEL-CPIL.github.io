"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

# Pre-compute power basis coefficients for Hermite polynomials H0, H4, H8, H12
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 98304x^10 + 1182720x^8 - 6635520x^6 + 15482880x^4 - 1451520x^2 + 665280

# Coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
H0_pow = np.array([1, 0, 0, 0, 0, 0, 0])
H4_pow = np.array([12, -48, 16, 0, 0, 0, 0])
H8_pow = np.array([1680, -13440, 11520, -3072, 256, 0, 0])
H12_pow = np.array([665280, -1451520, 15482880, -6635520, 1182720, -98304, 4096])

# Values at 0 for constraint P(0)=0
H0_0 = H0_pow[0]
H4_0 = H4_pow[0]
H8_0 = H8_pow[0]
H12_0 = H12_pow[0]

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    def calculate_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    # Pre-compute grid for fast vectorized evaluation
    # e^{-x^2/4} decays: at x=12, e^{-36} ~ 2e-16
    x_grid = np.linspace(-12, 12, 20000)
    x2_grid = x_grid * x_grid
    exp_grid = np.exp(-x2_grid / 4.0)
    # Precompute powers of x^2 for Horner evaluation
    x2_powers = [np.ones_like(x2_grid)]
    for i in range(1, 7):
        x2_powers.append(x2_powers[-1] * x2_grid)

    def eval_P_fast(c0, c4, c8, x2):
        """Evaluate P(x) using power basis - fast for single x^2 value."""
        c12 = calculate_c12(c0, c4, c8)
        coeff_total = c0 * H0_pow + c4 * H4_pow + c8 * H8_pow + c12 * H12_pow
        # Horner-like evaluation for polynomial in x^2
        result = coeff_total[6]
        for i in range(5, -1, -1):
            result = result * x2 + coeff_total[i]
        return result

    def objective_grid(coeffs, eps=0.0):
        """Fast vectorized objective using grid."""
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        coeff_total = c0 * H0_pow + c4 * H4_pow + c8 * H8_pow + c12 * H12_pow
        
        # Evaluate P on grid using Horner
        P = coeff_total[6] * x2_powers[6]
        for i in range(5, -1, -1):
            P = P * x2_grid + coeff_total[i] * x2_powers[i]
        
        if eps > 0:
            abs_P = np.sqrt(P * P + eps)
        else:
            abs_P = np.abs(P)
        
        integrand = abs_P * exp_grid
        return np.trapz(integrand, x_grid)

    def objective_quad(coeffs):
        """High-precision objective using adaptive quadrature."""
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        
        def integrand(x):
            P = eval_P_fast(c0, c4, c8, x * x)
            return np.abs(P) * np.exp(-x * x / 4.0)
            
        try:
            result, error = integrate.quad(
                integrand, -15, 15, limit=300, epsabs=1e-13, epsrel=1e-13
            )
            return result
        except Exception:
            return np.inf

    def smooth_objective(coeffs, eps=1e-4):
        """Smooth surrogate for |P| using sqrt(P^2 + eps)."""
        return objective_grid(coeffs, eps=eps)

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = x0.copy()
    best_val = objective_grid(x0)
    
    # Stage 1: Quick local refinement from seed using grid objective
    if time.time() < deadline - 100:
        try:
            res_quick = opt.minimize(
                lambda c: objective_grid(c), 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if res_quick.fun < best_val:
                best_val = res_quick.fun
                best_result = res_quick.x.copy()
        except Exception:
            pass

    # Stage 2: Global search using Differential Evolution on smooth grid objective
    if time.time() < deadline - 80:
        try:
            bounds = [(-3, 3), (-0.5, 0.5), (-0.005, 0.005)]
            de_res = opt.differential_evolution(
                lambda c: smooth_objective(c, 1e-3), 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=30,
                tol=1e-9, 
                mutation=(0.5, 1.5), 
                recombination=0.7,
                polish=False
            )
            if de_res.fun < best_val:
                grid_val = objective_grid(de_res.x)
                if grid_val < best_val:
                    best_val = grid_val
                    best_result = de_res.x.copy()
        except Exception:
            pass

    # Stage 3: Local refinement with decreasing epsilon for smoothness
    epsilons = [1e-3, 1e-5, 1e-7, 1e-9]
    for eps in epsilons:
        if time.time() > deadline - 40:
            break
        try:
            res_smooth = opt.minimize(
                lambda c: smooth_objective(c, eps), 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if res_smooth.fun < best_val:
                grid_val = objective_grid(res_smooth.x)
                if grid_val < best_val:
                    best_val = grid_val
                    best_result = res_smooth.x.copy()
        except Exception:
            pass

    # Stage 4: Try diverse starting points with quick optimization
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        candidates = [
            x0 + rng.normal(0, 0.1, 3),
            x0 + rng.normal(0, 0.5, 3),
            np.array([-1.0, 0.05, 0.0002]),
            np.array([-1.2, 0.03, 0.0004]),
            np.array([-1.1, 0.045, 0.00028]),
            np.array([-1.15, 0.038, 0.0003]),
            np.array([-1.08, 0.042, 0.00025]),
        ]
        for x_start in candidates:
            if time.time() > deadline - 15:
                break
            try:
                res_r = opt.minimize(
                    lambda c: objective_grid(c), 
                    x_start, 
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_result = res_r.x.copy()
            except Exception:
                pass

    # Stage 5: Final high-precision polish with quad
    if time.time() < deadline - 10:
        try:
            # First get a good estimate from grid
            grid_best = objective_grid(best_result)
            
            # Polish with quad
            res_final = opt.minimize(
                objective_quad, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_result = res_final.x.copy()
        except Exception:
            pass

    # Stage 6: One more quad polish if time remains
    if time.time() < deadline - 5:
        try:
            res_polish = opt.minimize(
                objective_quad, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_result = res_polish.x.copy()
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
