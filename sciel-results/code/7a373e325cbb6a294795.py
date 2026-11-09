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
# H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
H0_0 = H0(0)
H4_0 = H4(0)
H8_0 = H8(0)
H12_0 = H12(0)

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    def calculate_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    # Use a pre-computed grid for fast evaluation (vectorized)
    # The integrand decays as e^{-x^2/4}. At x=15, e^{-56.25} ~ 4e-25.
    # 20000 points over [-15, 15] gives dx=0.0015.
    # This should be accurate enough for optimization.
    x_grid = np.linspace(-15, 15, 20000)
    exp_term = np.exp(-x_grid**2 / 4.0)
    # Pre-compute Hermite polynomial values on the grid
    H0_vals = H0(x_grid)
    H4_vals = H4(x_grid)
    H8_vals = H8(x_grid)
    H12_vals = H12(x_grid)
    
    # dx for trapezoidal rule
    dx = x_grid[1] - x_grid[0]

    def objective(coeffs):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        
        # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        P = c0 * H0_vals + c4 * H4_vals + c8 * H8_vals + c12 * H12_vals
        
        # Integral of |P(x)| e^{-x^2/4}
        integrand = np.abs(P) * exp_term
        integral = np.sum(integrand) * dx # Rectangle rule approximation, fast
        
        return integral

    # Smooth surrogate using soft-min to handle non-differentiability of |P|
    # We want to minimize the integral of |P(x)| e^{-x^2/4}
    # This is equivalent to minimizing t subject to t >= |P(x_i)| e^{-x_i^2/4} * dx for all i
    # Or we can use a smooth approximation of |P|: sqrt(P^2 + eps^2)
    # Or use the epigraph formulation with scipy.optimize.minimize and constraints
    # Let's try a smooth approximation first: replace |P| with sqrt(P^2 + eps^2)
    
    def objective_smooth(coeffs, eps=1e-6):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        
        P = c0 * H0_vals + c4 * H4_vals + c8 * H8_vals + c12 * H12_vals
        
        # Smooth absolute value: sqrt(P^2 + eps^2)
        smooth_abs = np.sqrt(P**2 + eps**2)
        integrand = smooth_abs * exp_term
        integral = np.sum(integrand) * dx
        
        return integral

    # High-precision objective for final polish
    def objective_precise(coeffs):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        
        def integrand(x):
            P = c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)
            return np.abs(P) * np.exp(-x**2 / 4.0)
            
        result, error = integrate.quad(
            integrand, -15, 15, limit=200, epsabs=1e-12, epsrel=1e-12
        )
        return result

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = x0.copy()
    best_val = objective(x0)
    
    # 1. Differential Evolution for global search (using fast objective)
    # DE is robust to non-smoothness of |P| and can find global minima.
    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
            de_res = opt.differential_evolution(
                objective, bounds, seed=42, maxiter=100, popsize=20,
                tol=1e-10, mutation=(0.5, 1.5), recombination=0.7,
                polish=False  # We will polish manually
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            pass

    # 2. Local refinement with Nelder-Mead from best found so far
    if best_result is not None and time.time() < deadline - 30:
        try:
            res2 = opt.minimize(objective, best_result, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass

    # 3. Try a few diverse starting points with local optimization
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        # Try points near the seed and some random ones
        candidates = [
            x0 + rng.normal(0, 0.1, 3),
            x0 + rng.normal(0, 0.5, 3),
            np.array([-1.0, 0.05, 0.0002]),
            np.array([-1.2, 0.03, 0.0004]),
            np.array([-1.1, 0.045, 0.00028]),
            # Some more structured guesses
            np.array([-1.15, 0.035, 0.0003]),
            np.array([-1.05, 0.045, 0.00015]),
        ]
        for x_start in candidates:
            if time.time() > deadline - 10:
                break
            try:
                res_r = opt.minimize(objective, x_start, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_result = res_r.x.copy()
            except Exception:
                pass

    # 4. Use smooth surrogate with gradient-based optimization for finer tuning
    if time.time() < deadline - 10:
        try:
            # Use L-BFGS-B with smooth objective
            bounds_smooth = [(-5, 5), (-1, 1), (-0.01, 0.01)]
            res_smooth = opt.minimize(objective_smooth, best_result, method='L-BFGS-B',
                                     bounds=bounds_smooth,
                                     options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res_smooth.fun < best_val:
                best_val = res_smooth.fun
                best_result = res_smooth.x.copy()
        except Exception:
            pass

    # 5. Final high-precision polish
    if time.time() < deadline - 5:
        try:
            # Use precise objective for final steps
            res_final = opt.minimize(objective_precise, best_result, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_result = res_final.x.copy()
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
