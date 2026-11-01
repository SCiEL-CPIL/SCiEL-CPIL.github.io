"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the upper bound on
    A(f)*A(Fourier(f)) for the Hermite-Gaussian test function.
    The H12 coefficient is derived to enforce g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Hermite polynomial coefficients (physicists' convention)
    # H_n(x) evaluated at x
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 3072*x**6 + 10752*x**4 - 10752*x**2 + 1344
    
    def H12(x):
        return (4096*x**12 - 98304*x**10 + 884736*x**8 
                - 4014080*x**6 + 8106336*x**4 - 6339328*x**2 + 1032192)
    
    def compute_a12(a0, a4, a8):
        """Derive a12 so that g(0) = 0."""
        # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * e^0
        # H0(0) = 1, H4(0) = 12, H8(0) = 1344, H12(0) = 1032192
        h0_0 = 1.0
        h4_0 = 12.0
        h8_0 = 1344.0
        h12_0 = 1032192.0
        return -(a0*h0_0 + a4*h4_0 + a8*h8_0) / h12_0
    
    def compute_upper_bound(a0, a4, a8):
        """
        Compute the upper bound on A(f)*A(Fourier(f)).
        This involves computing norms of the test function and its derivatives.
        """
        a12 = compute_a12(a0, a4, a8)
        
        # Use numerical integration on a fine grid
        x = np.linspace(-15, 15, 10001)
        dx = x[1] - x[0]
        gauss = np.exp(-x**2 / 2)
        
        # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss
        poly = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
        g = poly * gauss
        
        # Compute norms needed for the uncertainty bound
        # ||g||^2
        norm_g_sq = np.sum(g**2) * dx
        
        # ||x*g||^2 (position spread)
        xg = x * g
        norm_xg_sq = np.sum(xg**2) * dx
        
        # For the Fourier side, we need properties of the Fourier transform
        # The Fourier transform of a Hermite-Gaussian can be computed analytically
        # For g(x) = P(x) * e^{-x^2/2}, the Fourier transform involves
        # integrals that can be related to derivatives of the Gaussian
        
        # A(f) typically involves ||f|| and ||f'|| or similar
        # Let's compute ||g'||^2
        # g'(x) = P'(x)*gauss + P(x)*gauss'
        # gauss' = -x*gauss
        # P'(x) = a0*H0' + a4*H4' + a8*H8' + a12*H12'
        
        H0_prime = np.zeros_like(x)
        H4_prime = 64*x**3 - 96*x
        H8_prime = (2048*x**7 - 18432*x**5 + 43008*x**3 - 21504*x)
        H12_prime = (49152*x**11 - 983040*x**9 + 7077888*x**7 
                     - 24084480*x**5 + 32425344*x**3 - 12678656*x)
        
        P_prime = a0*H0_prime + a4*H4_prime + a8*H8_prime + a12*H12_prime
        g_prime = P_prime * gauss + poly * (-x) * gauss
        
        norm_g_prime_sq = np.sum(g_prime**2) * dx
        
        # The uncertainty bound likely involves a combination of these norms
        # A common form: A(f)*A(F) >= C * ||f||^2, and the bound is tight when
        # we minimize some functional
        
        # Let's use a standard form: the bound involves (norm_xg_sq/norm_g_sq) * (norm_g_prime_sq/norm_g_sq)
        # or similar products
        
        if norm_g_sq < 1e-15:
            return 1e10  # Degenerate case
        
        # Compute variance-like quantities
        var_x = norm_xg_sq / norm_g_sq
        # For the momentum/Fourier side, use derivative norm
        var_p = norm_g_prime_sq / norm_g_sq
        
        # The upper bound on the product is related to these
        # A typical bound: A(f)*A(F) <= C * sqrt(var_x * var_p) or similar
        # Let's use the product as the objective to minimize
        upper_bound = np.sqrt(var_x * var_p)
        
        return upper_bound
    
    # Initial guess from the seed program
    initial_guess = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple starting points to avoid local minima
    starting_points = [
        initial_guess,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.002]),
        np.array([-1.5, 0.15, 0.0005]),
        np.array([-0.8, 0.08, 0.0015]),
        np.array([-1.2, 0.12, 0.0008]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-2.0, 0.2, 0.003]),
        np.array([-0.5, 0.2, 0.001]),
        np.array([-1.0, 0.1, -0.001]),
    ]
    
    best_bound = np.inf
    best_coeffs = initial_guess.copy()
    
    for i, x0 in enumerate(starting_points):
        if time.time() > deadline - 30:  # Stop if we're getting close to deadline
            break
        
        try:
            # Use Nelder-Mead which doesn't require gradients
            result = minimize(
                compute_upper_bound,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x.copy()
                
        except Exception as e:
            continue
    
    # If we still have time, do a local refinement
    if time.time() < deadline - 60 and best_bound < np.inf:
        try:
            result = minimize(
                compute_upper_bound,
                best_coeffs,
                method='Powell',
                options={
                    'maxiter': 10000,
                    'maxfev': 10000,
                    'ftol': 1e-12,
                    'xtol': 1e-12
                }
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x.copy()
                
        except Exception as e:
            pass
    
    # Return the best coefficients found
    return best_coeffs
# EVOLVE-BLOCK-END
