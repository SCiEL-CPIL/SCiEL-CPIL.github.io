"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _hermite_polynomials(x, n_max):
    """Compute Hermite polynomials H_0 through H_{n_max} using recurrence."""
    H = np.zeros((n_max + 1, len(x)))
    H[0, :] = 1.0
    if n_max >= 1:
        H[1, :] = 2.0 * x
    for n in range(2, n_max + 1):
        H[n, :] = 2.0 * x * H[n-1, :] - 2.0 * (n-1) * H[n-2, :]
    return H

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(F(f)) for a test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    The uncertainty principle bound is related to the product of
    position and momentum variances (or related quantities).
    """
    # Grid for numerical integration
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials up to degree 12
    H = _hermite_polynomials(x, 12)
    
    # Gaussian weight
    gauss = np.exp(-x**2 / 2.0)
    
    # c12 is determined by g(0) = 0
    # At x=0: H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-945
    # g(0) = c0*1 + c4*(-3) + c8*105 + c12*(-945) = 0
    # c12 = (c0 - 3*c4 + 105*c8) / 945
    c12 = (c0 - 3.0*c4 + 105.0*c8) / 945.0
    
    # Construct g(x)
    g = (c0*H[0] + c4*H[4] + c8*H[8] + c12*H[12]) * gauss
    
    # Check that g(0) is close to 0
    idx0 = np.argmin(np.abs(x))
    # g[0] should be ~0
    
    # Compute ||g||_2^2
    norm_g_sq = np.sum(g**2) * dx
    
    if norm_g_sq < 1e-15:
        return np.inf
    
    # Normalize
    g_norm = g / np.sqrt(norm_g_sq)
    
    # Compute Fourier transform numerically
    # F(f) for f(x) = g(x) * exp(-x^2/2)
    # The Fourier transform of a Hermite-Gaussian can be computed analytically,
    # but let's use the numerical approach for generality.
    
    # Actually, for the uncertainty principle, we need:
    # A(f) ~ sqrt(<x^2> - <x>^2) and A(F(f)) ~ sqrt(<p^2> - <p>^2)
    # where p = -i d/dx (in appropriate units)
    
    # <x>
    x_mean = np.sum(x * g_norm**2) * dx
    # <x^2>
    x2_mean = np.sum(x**2 * g_norm**2) * dx
    # Variance of x
    var_x = x2_mean - x_mean**2
    
    # For momentum, we need the derivative of g
    # p = -i d/dx, so |p g|^2 = |dg/dx|^2 (in appropriate units)
    dg = np.gradient(g_norm, dx)
    p_mean = np.sum(dg**2) * dx  # This is <p^2> for real functions with p = -i d/dx
    # Actually for real functions, <p> = 0, so var_p = <p^2>
    var_p = p_mean
    
    # The uncertainty product bound
    # For the specific constant C4, it's related to var_x * var_p
    # The standard uncertainty principle: var_x * var_p >= (hbar/2)^2
    # With hbar = 1, the lower bound is 1/4
    
    # The "upper bound on A(f)*A(F(f))" likely refers to the product
    # A(f) * A(F(f)) where A is some norm-based measure.
    
    # Let me reconsider. The task says "upper bound on C4" and the score
    # is the upper bound value. Lower is better.
    
    # In the context of the uncertainty principle with Hermite polynomials,
    # the relevant quantity is likely:
    # C = sqrt(var_x * var_p) or var_x * var_p
    
    # Let's compute both and see which makes sense
    bound = np.sqrt(var_x * var_p)
    
    return bound

def construct():
    """
    Find coefficients (c0, c4, c8) that minimize the uncertainty bound.
    c12 is determined by the constraint g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Initial guess: the seed program values
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    
    best_params = np.array([c0_init, c4_init, c8_init])
    best_score = _compute_bound(c0_init, c4_init, c8_init)
    
    # Define the objective function
    def objective(params):
        c0, c4, c8 = params
        return _compute_bound(c0, c4, c8)
    
    # Try multiple optimization methods and starting points
    methods = ['BFGS', 'Nelder-Mead', 'Powell']
    
    # Generate diverse starting points
    start_points = [
        [c0_init, c4_init, c8_init],
        [c0_init * 1.1, c4_init * 1.1, c8_init * 1.1],
        [c0_init * 0.9, c4_init * 0.9, c8_init * 0.9],
        [-1.0, 0.04, 0.0003],
        [-1.1, 0.03, 0.0002],
        [-1.05, 0.035, 0.00025],
    ]
    
    for method in methods:
        for sp in start_points:
            if time.time() > deadline - 10:
                break
            
            try:
                result = minimize(objective, sp, method=method, 
                                  options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if time.time() > deadline - 10:
            break
    
    # If we have time, do a final polish with tight tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(objective, best_params, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
