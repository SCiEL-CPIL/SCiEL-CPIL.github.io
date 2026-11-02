"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time


def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s limit
    
    # Hermite polynomials (physicists' convention)
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    
    def H12(x):
        return 4096*x**12 - 73728*x**10 + 580608*x**8 - 2419200*x**6 + 5443200*x**4 - 5443200*x**2 + 2032128
    
    def g(x, c0, c4, c8):
        c12 = -(c0 + 12*c4 + 1680*c8) / 2032128.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f(x, c0, c4, c8):
        return g(x, c0, c4, c8) * np.exp(-x**2 / 2.0)
    
    def compute_uncertainty_product(c0, c4, c8):
        """Compute Δx * Δp for the test function. Since f is Fourier-invariant, Δp = Δx."""
        # L2 norm squared
        norm_sq, _ = quad(lambda x: f(x, c0, c4, c8)**2, -np.inf, np.inf, limit=200)
        if norm_sq <= 0 or np.isnan(norm_sq):
            return np.inf
        
        # Second moment
        moment_x, _ = quad(lambda x: x**2 * f(x, c0, c4, c8)**2, -np.inf, np.inf, limit=200)
        if np.isnan(moment_x) or moment_x < 0:
            return np.inf
        
        var_x = moment_x / norm_sq
        if var_x < 0:
            return np.inf
        
        # Δx * Δp = var_x since Δp = Δx for Fourier-invariant functions
        return var_x
    
    # Seed values from Goncalves et al. 2017
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Evaluate seed
    seed_val = compute_uncertainty_product(*seed_coeffs)
    
    best_coeffs = seed_coeffs.copy()
    best_val = seed_val
    
    # If seed evaluation failed, use a safe fallback
    if not np.isfinite(best_val) or best_val <= 0:
        best_coeffs = np.array([0.0, 0.0, 0.0])
        best_val = np.inf
    
    # Try local optimization around seed with tight bounds
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(
                lambda c: compute_uncertainty_product(*c),
                seed_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 200, 'adaptive': True}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Try a few carefully chosen perturbations
    if time.time() - start_time < deadline - 10:
        np.random.seed(12345)
        for scale in [0.01, 0.005, 0.001]:
            if time.time() - start_time > deadline - 5:
                break
            for _ in range(10):
                if time.time() - start_time > deadline - 5:
                    break
                noise = np.random.normal(0, scale, 3)
                x_try = best_coeffs + noise
                try:
                    val = compute_uncertainty_product(*x_try)
                    if np.isfinite(val) and val < best_val:
                        best_val = val
                        best_coeffs = x_try.copy()
                        # Local polish
                        res = minimize(
                            lambda c: compute_uncertainty_product(*c),
                            x_try,
                            method='Nelder-Mead',
                            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 100, 'adaptive': True}
                        )
                        if np.isfinite(res.fun) and res.fun < best_val:
                            best_val = res.fun
                            best_coeffs = res.x.copy()
                except Exception:
                    pass
    
    # Final polish
    if time.time() - start_time < deadline - 2:
        try:
            res = minimize(
                lambda c: compute_uncertainty_product(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 500, 'adaptive': True}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
