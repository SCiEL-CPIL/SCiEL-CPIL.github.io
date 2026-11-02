"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F) for given coefficients.
    
    Uses exact Hermite-Gaussian Fourier properties and dense grid evaluation.
    """
    # H_n(0) values for physicists' Hermite polynomials
    # H_0(0)=1, H_4(0)=12, H_8(0)=1680, H_12(0)=665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    def _g(x):
        """Evaluate g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)"""
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x6 * x6
        H0 = np.ones_like(x)
        H4 = 16.0*x4 - 48.0*x2 + 12.0
        H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
        H12 = (4096.0*x12 - 245760.0*x10 + 5529600.0*x8 
               - 45158400.0*x6 + 164430720.0*x4 
               - 222772500.0*x2 + 665280.0)
        poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
        return poly * np.exp(-0.5 * x2)
    
    # Dense grid to find sup |x * g(x)|
    # The function decays like exp(-x^2/2) times a polynomial, so it's concentrated
    # in roughly |x| < 10. Use a fine grid.
    N_grid = 20000
    x = np.linspace(-12.0, 12.0, N_grid)
    xg = x * _g(x)
    Af = np.max(np.abs(xg))
    
    # Refine around the maximum
    idx = np.argmax(np.abs(xg))
    if 0 < idx < N_grid - 1:
        # Local refinement
        x_lo = x[idx-1]
        x_hi = x[idx+1]
        x_fine = np.linspace(x_lo, x_hi, 1000)
        xg_fine = x_fine * _g(x_fine)
        Af_local = np.max(np.abs(xg_fine))
        if Af_local > Af:
            Af = Af_local
    
    # The Fourier transform: G(k) = sqrt(2*pi) * g(k) since (-i)^n = 1 for n in {0,4,8,12}
    # So A(F) = sqrt(2*pi) * A(f)
    # Product = sqrt(2*pi) * Af^2
    bound = math.sqrt(2.0 * math.pi) * Af * Af
    
    return bound


def _smooth_bound(c0, c4, c8, beta=50.0):
    """Smooth surrogate for the bound using soft-max."""
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    def _g(x):
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        x10 = x8 * x2
        x12 = x6 * x6
        H0 = np.ones_like(x)
        H4 = 16.0*x4 - 48.0*x2 + 12.0
        H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
        H12 = (4096.0*x12 - 245760.0*x10 + 5529600.0*x8 
               - 45158400.0*x6 + 164430720.0*x4 
               - 222772500.0*x2 + 665280.0)
        poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
        return poly * np.exp(-0.5 * x2)
    
    N_grid = 5000
    x = np.linspace(-10.0, 10.0, N_grid)
    xg = x * _g(x)
    abs_xg = np.abs(xg)
    
    # Soft-max approximation
    max_val = np.max(abs_xg)
    # Use log-sum-exp style: max + (1/beta)*log(sum(exp(beta*(x_i - max))))
    shifted = beta * (abs_xg - max_val)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    Af_soft = max_val + log_sum / beta
    
    bound = math.sqrt(2.0 * math.pi) * Af_soft * Af_soft
    return bound


def construct():
    """Find optimal coefficients by numerical optimization."""
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s limit
    
    # Initial guess
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    def objective(coeffs):
        try:
            return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
        except:
            return 1e10
    
    def smooth_objective(coeffs):
        try:
            return _smooth_bound(coeffs[0], coeffs[1], coeffs[2])
        except:
            return 1e10
    
    best_val = objective(x0)
    best_x = x0.copy()
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            de_res = differential_evolution(
                smooth_objective, bounds,
                maxiter=100, popsize=15, seed=42,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.9,
                polish=False, update_interval=10
            )
            if de_res.fun < best_val:
                best_val = objective(de_res.x)
                best_x = de_res.x.copy()
        except:
            pass
    
    # Phase 2: Nelder-Mead refinement from multiple starts
    start_points = [
        best_x,
        x0,
        np.array([-1.1, 0.04, 0.001]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.15, 0.035, 0.0008]),
        np.array([-1.05, 0.045, 0.0009]),
        np.array([-1.25, 0.025, 0.0004]),
        np.array([-0.95, 0.055, 0.0011]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Final verification with high-resolution grid
    if time.time() < deadline - 5:
        try:
            # Use even finer grid for final evaluation
            H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 1680.0, 665280.0
            c12 = -(best_x[0] * H0_0 + best_x[1] * H4_0 + best_x[2] * H8_0) / H12_0
            
            def _g_fine(x):
                x2 = x * x
                x4 = x2 * x2
                x6 = x4 * x2
                x8 = x4 * x4
                x10 = x8 * x2
                x12 = x6 * x6
                H0 = np.ones_like(x)
                H4 = 16.0*x4 - 48.0*x2 + 12.0
                H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
                H12 = (4096.0*x12 - 245760.0*x10 + 5529600.0*x8 
                       - 45158400.0*x6 + 164430720.0*x4 
                       - 222772500.0*x2 + 665280.0)
                poly = best_x[0]*H0 + best_x[1]*H4 + best_x[2]*H8 + c12*H12
                return poly * np.exp(-0.5 * x2)
            
            N_fine = 100000
            x_fine = np.linspace(-15.0, 15.0, N_fine)
            xg_fine = x_fine * _g_fine(x_fine)
            Af_fine = np.max(np.abs(xg_fine))
            final_bound = math.sqrt(2.0 * math.pi) * Af_fine * Af_fine
            
            if final_bound < best_val:
                best_val = final_bound
        except:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
