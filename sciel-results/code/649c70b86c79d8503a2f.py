import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    start_time = time.time()
    deadline = 180.0 # Safety margin before 210s kill
    
    # Known state-of-the-art coefficients (Goncalves et al. 2017)
    # These are coefficients for H0, H4, H8. H12 is derived to satisfy g(0)=0.
    c0_known = -113.0 / 100.0
    c4_known = 1.0 / 25.0
    c8_known = 1.0 / 3240.0
    
    best_coeffs = np.array([c0_known, c4_known, c8_known])
    best_score = float('inf') # We want to minimize the bound
    
    # Helper to calculate c12 from c0, c4, c8 such that g(0) = 0
    # H_n(0) values for even n:
    # H_0(0) = 1
    # H_4(0) = -3
    # H_8(0) = 105
    # H_12(0) = -10395
    # g(0) = c0*1 + c4*(-3) + c8*105 + c12*(-10395) = 0
    # c12 = (c0 - 3*c4 + 105*c8) / 10395
    def get_c12(c0, c4, c8):
        return (c0 - 3.0*c4 + 105.0*c8) / 10395.0

    # Objective function: Calculate the uncertainty product for the given coefficients
    # We approximate the norms using numerical integration on a sufficient grid
    # g(x) = exp(-x^2/2) * P(x)
    # Fourier transform of exp(-x^2/2) P(x) is exp(-k^2/2) Q(k) where Q is related to P.
    # Specifically, if P(x) = sum c_n H_n(x), then \hat{g}(k) = (-i)^n sum c_n H_n(k) exp(-k^2/2) ?
    # Actually, \mathcal{F}[e^{-x^2/2} H_n(x)] = (-i)^n e^{-k^2/2} H_n(k).
    # So if g(x) = e^{-x^2/2} \sum c_n H_n(x), then \hat{g}(k) = e^{-k^2/2} \sum c_n (-i)^n H_n(k).
    # Since we only have even n (0,4,8,12), (-i)^n = 1 for all these n.
    # So \hat{g}(k) = e^{-k^2/2} \sum c_n H_n(k).
    # This means g and \hat{g} have the same form with the same coefficients!
    # Thus ||x g||_2 = ||x \hat{g}||_2 and ||g||_2 = ||\hat{g}||_2.
    # Uncertainty product U = ||x g||_2 ||\hat{g}||_2 = ||x g||_2 ||g||_2.
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Create a grid for numerical integration
        # The function decays as exp(-x^2/2) * poly(x).
        # A range of [-10, 10] should be sufficient for high precision.
        x = np.linspace(-10, 10, 10000)
        dx = x[1] - x[0]
        
        # Calculate P(x)
        H0 = eval_hermite(0, x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        g = np.exp(-x**2 / 2.0) * P
        
        # Calculate ||g||_2^2 and ||x g||_2^2
        # Note: Standard Hermite polynomials H_n are defined such that
        # \int_{-\infty}^{\infty} e^{-x^2} H_n(x) H_m(x) dx = \sqrt{\pi} 2^n n! \delta_{nm}
        # Our basis is e^{-x^2/2} H_n(x).
        # Let's just integrate numerically to be safe and consistent.
        
        g2 = g**2
        norm_g_sq = np.sum(g2) * dx
        norm_xg_sq = np.sum((x * g)**2) * dx
        
        # Avoid log(0)
        if norm_g_sq <= 1e-15:
            return 1e100
            
        uncertainty_product = np.sqrt(norm_xg_sq * norm_g_sq)
        
        # The task mentions "upper bound on A(f)*A(Fourier(f))".
        # Usually, the constant C is defined such that A(f)A(\hat{f}) >= C ||f||^2.
        # If we are minimizing the "upper bound" of the product for a specific class, 
        # we are effectively minimizing the uncertainty product of the test function.
        return uncertainty_product

    # Initial evaluation
    initial_val = objective(best_coeffs)
    best_score = initial_val
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return best_coeffs

    # Optimization
    # We try to refine the coefficients.
    # Use Nelder-Mead as it doesn't require gradients and is robust for low dimensions.
    try:
        # Bounds for coefficients to keep the function well-behaved
        # The coefficients shouldn't be too large or the polynomial part will dominate
        # and the uncertainty product will increase.
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        
        # Run optimization
        result = minimize(
            objective, 
            best_coeffs, 
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
            
    except Exception as e:
        # If optimization fails, stick with the known best
        pass

    # Check time again
    if time.time() - start_time > deadline - 5:
        return best_coeffs

    # Return the best coefficients found
    # The interface expects the three coefficients (H0, H4, H8)
    return best_coeffs
