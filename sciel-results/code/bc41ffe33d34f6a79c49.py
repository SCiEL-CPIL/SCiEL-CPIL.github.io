"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import factorial
import time
import math


def _compute_bound(c0, c4, c8, N=200):
    """Compute the upper bound on A(f)*A(F) for given coefficients.
    
    Uses exact Hermite-Gaussian Fourier properties and Gauss-Hermite quadrature.
    """
    # Coefficients for H_n(0) to determine c12 such that g(0)=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Use Gauss-Hermite quadrature with sufficient points
    x_gh, w_gh = np.polynomial.hermite.hermgauss(N)
    
    # Compute Hermite polynomials at quadrature points
    x2 = x_gh ** 2
    H0 = np.ones_like(x_gh)
    H4 = 16*x2*x2 - 48*x2 + 12
    H8 = 256*x2*x2*x2*x2 - 4480*x2*x2*x2 + 26880*x2*x2 - 6720*x2 + 1680
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = 4096*x12 - 129024*x10 + 1761280*x8 - 11942400*x6 + 37158400*x4 - 54432000*x2 + 47900160
    
    # g(x) at quadrature points
    g_vals = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x_gh**2 / 2.0)
    
    # A(f) = sup |x * g(x)|
    xg = x_gh * g_vals
    Af = np.max(np.abs(xg))
    
    # The product is sqrt(2*pi) * Af^2
    bound = math.sqrt(2 * math.pi) * Af * Af
    
    return bound


def _compute_smooth_bound(coeffs, N=200, beta=50.0):
    """Compute a smooth approximation of the bound using soft-max.
    
    This makes the objective differentiable everywhere, helping gradient-based methods.
    """
    c0, c4, c8 = coeffs
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    x_gh, w_gh = np.polynomial.hermite.hermgauss(N)
    
    x2 = x_gh ** 2
    H0 = np.ones_like(x_gh)
    H4 = 16*x2*x2 - 48*x2 + 12
    H8 = 256*x2*x2*x2*x2 - 4480*x2*x2*x2 + 26880*x2*x2 - 6720*x2 + 1680
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = 4096*x12 - 129024*x10 + 1761280*x8 - 11942400*x6 + 37158400*x4 - 54432000*x2 + 47900160
    
    g_vals = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x_gh**2 / 2.0)
    
    xg = np.abs(x_gh * g_vals)
    
    # Soft-max approximation: (1/beta) * log(sum(exp(beta * x)))
    # For large beta, this approaches max(x)
    scaled = beta * xg
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    Af_smooth = log_sum_exp / beta
    
    bound = math.sqrt(2 * math.pi) * Af_smooth * Af_smooth
    return bound


def _compute_bound_grad(coeffs, N=200):
    """Compute bound and gradient using finite differences."""
    c0, c4, c8 = coeffs
    f0 = _compute_bound(c0, c4, c8, N)
    
    h = 1e-8
    dfdc0 = (_compute_bound(c0+h, c4, c8, N) - _compute_bound(c0-h, c4, c8, N)) / (2*h)
    dfdc4 = (_compute_bound(c0, c4+h, c8, N) - _compute_bound(c0, c4-h, c8, N)) / (2*h)
    dfdc8 = (_compute_bound(c0, c4, c8+h, N) - _compute_bound(c0, c4-h, c8, N)) / (2*h)
    
    return f0, np.array([dfdc0, dfdc4, dfdc8])


def construct():
    """Find optimal coefficients by numerical optimization."""
    start_time = time.time()
    deadline = 175.0  # Safety margin before 210s limit
    
    # Initial guess from prior state of the art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    def objective(coeffs):
        try:
            return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
        except:
            return 1e10
    
    def objective_smooth(coeffs):
        try:
            return _compute_smooth_bound(coeffs)
        except:
            return 1e10
    
    best_val = objective(x0)
    best_x = x0.copy()
    
    from scipy.optimize import minimize, differential_evolution
    
    # Phase 0: Differential evolution for global search (time-limited)
    if time.time() < deadline - 80:
        try:
            bounds = [(-2.0, 0.0), (0.0, 0.1), (0.0, 0.01)]
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-6, mut=(0.5, 1.5), recomb=0.8, polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except:
            pass
    
    # Phase 1: Nelder-Mead from multiple start points
    start_points = [
        x0,
        np.array([-1.1, 0.04, 0.001]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.15, 0.035, 0.0008]),
        np.array([-1.05, 0.045, 0.0012]),
        np.array([-1.3, 0.025, 0.0003]),
        np.array([-0.9, 0.06, 0.0015]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 2: L-BFGS-B with gradient for higher precision
    if time.time() < deadline - 25:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', 
                          jac=lambda x: _compute_bound_grad(x)[1],
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 3: Use smooth objective for gradient-based refinement
    if time.time() < deadline - 15:
        try:
            def smooth_jac(x):
                h = 1e-7
                f0 = objective_smooth(x)
                grad = np.zeros(3)
                for i in range(3):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += h
                    xm[i] -= h
                    grad[i] = (objective_smooth(xp) - objective_smooth(xm)) / (2*h)
                return grad
            
            res = minimize(objective_smooth, best_x, method='L-BFGS-B',
                          jac=smooth_jac,
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            # Verify with exact objective
            val_exact = objective(res.x)
            if val_exact < best_val:
                best_val = val_exact
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 4: Final polish with Nelder-Mead at very tight tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Time check: if we're close to deadline, return immediately
    if time.time() > deadline - 2:
        return np.array(best_x)
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
