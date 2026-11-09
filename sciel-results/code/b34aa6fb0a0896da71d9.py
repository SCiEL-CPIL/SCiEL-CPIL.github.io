"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import scipy.integrate as integrate
import time

# Pre-compute Hermite polynomials
# Physicists' Hermite polynomials
H0 = sp.hermite(0)
H4 = sp.hermite(4)
H8 = sp.hermite(8)
H12 = sp.hermite(12)

# Values at 0 for constraint P(0)=0
H0_0 = H0(0)
H4_0 = H4(0)
H8_0 = H8(0)
H12_0 = H12(0)

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    def calculate_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    # Pre-computed grid for fast evaluation
    x_grid = np.linspace(-12, 12, 15000)
    exp_term = np.exp(-x_grid**2 / 4.0)
    H0_vals = H0(x_grid)
    H4_vals = H4(x_grid)
    H8_vals = H8(x_grid)
    H12_vals = H12(x_grid)
    dx = x_grid[1] - x_grid[0]

    def P_from_coeffs(c0, c4, c8, c12):
        return c0 * H0_vals + c4 * H4_vals + c8 * H8_vals + c12 * H12_vals

    def objective_exact(coeffs):
        """Exact objective using |P| on grid."""
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        P = P_from_coeffs(c0, c4, c8, c12)
        integrand = np.abs(P) * exp_term
        return np.sum(integrand) * dx

    def objective_smooth(coeffs, eps=1e-6):
        """Smooth surrogate: sqrt(P^2 + eps^2) instead of |P|."""
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        P = P_from_coeffs(c0, c4, c8, c12)
        integrand = np.sqrt(P**2 + eps**2) * exp_term
        return np.sum(integrand) * dx

    def objective_precise(coeffs):
        """High-precision objective using adaptive quadrature."""
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        
        def integrand(x):
            P = c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)
            return abs(P) * np.exp(-x**2 / 4.0)
            
        result, error = integrate.quad(
            integrand, -12, 12, limit=300, epsabs=1e-14, epsrel=1e-14
        )
        return result

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = x0.copy()
    best_val = objective_exact(x0)

    # Phase 1: Global search with DE using smooth objective
    # Tighter bounds based on seed: c0 ~ -1.13, c4 ~ 0.04, c8 ~ 3e-4
    if time.time() < deadline - 80:
        try:
            bounds = [(-2.0, -0.5), (-0.05, 0.15), (-0.001, 0.002)]
            de_res = opt.differential_evolution(
                objective_smooth, bounds, seed=42, maxiter=150, popsize=25,
                tol=1e-12, mutation=(0.5, 1.5), recombination=0.8,
                polish=False, updating='deferred'
            )
            if de_res.fun < best_val:
                # Verify with exact objective
                exact_val = objective_exact(de_res.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_result = de_res.x.copy()
        except Exception:
            pass

    # Phase 2: Local refinement with Nelder-Mead on exact objective
    if time.time() < deadline - 40:
        try:
            res2 = opt.minimize(objective_exact, best_result, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Try diverse starting points with local optimization
    if time.time() < deadline - 25:
        rng = np.random.default_rng(123)
        # Structured candidates around the seed and DE result
        candidates = [
            x0 + rng.normal(0, 0.05, 3),
            x0 + rng.normal(0, 0.1, 3),
            x0 + rng.normal(0, 0.3, 3),
            np.array([-1.10, 0.042, 0.00029]),
            np.array([-1.15, 0.038, 0.00032]),
            np.array([-1.08, 0.045, 0.00027]),
            np.array([-1.12, 0.041, 0.00030]),
            np.array([-1.14, 0.039, 0.00031]),
        ]
        # Also add perturbations around best_result
        candidates.append(best_result + rng.normal(0, 0.02, 3))
        candidates.append(best_result + rng.normal(0, 0.05, 3))
        
        for x_start in candidates:
            if time.time() > deadline - 15:
                break
            try:
                res_r = opt.minimize(objective_exact, x_start, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13})
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_result = res_r.x.copy()
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res_final = opt.minimize(objective_precise, best_result, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-13, 'fatol': 1e-15})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_result = res_final.x.copy()
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
