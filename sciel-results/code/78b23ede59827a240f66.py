import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time

# Set a deadline
START_TIME = time.time()
TIME_LIMIT = 210  # seconds

def get_hermite(n):
    # Returns the physicist's Hermite polynomial H_n
    return hermite(n)

def compute_c12(c0, c4, c8):
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-290304
    # c0*1 + c4*(-12) + c8*1680 + c12*(-290304) = 0
    return (c0 - 12*c4 + 1680*c8) / 290304

def objective(coeffs):
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return np.inf
    
    c0, c4, c8 = coeffs
    c12 = compute_c12(c0, c4, c8)
    
    # Define g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    # We need to compute the uncertainty product.
    # Let's use numerical integration on a sufficiently large grid.
    x = np.linspace(-15, 15, 2000)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = hermite(4)(x)
    H8 = hermite(8)(x)
    H12 = hermite(12)(x)
    
    poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
    g = poly * np.exp(-x**2 / 2)
    
    # Norm squared
    norm_sq = np.trapz(g**2, x)
    if norm_sq < 1e-12:
        return 1e12
        
    # Second moment in x
    x2_moment = np.trapz(x**2 * g**2, x)
    
    # Derivative g'(x)
    # g'(x) = (poly' - x*poly) * exp(-x^2/2)
    H0_d = np.zeros_like(x)
    H4_d = hermite(4)(x).derivative() if hasattr(hermite(4)(x), 'derivative') else 4*hermite(3)(x)
    H8_d = 8*hermite(7)(x)
    H12_d = 12*hermite(11)(x)
    
    poly_d = c0*H0_d + c4*H4_d + c8*H8_d + c12*H12_d
    g_prime = (poly_d - x*poly) * np.exp(-x**2 / 2)
    
    # Second moment in frequency space is proportional to integral of |g'|^2
    # Using standard FT: F[f'] = i*xi*F[f]. So int |F[f]|^2 xi^2 dx = int |f'|^2 dx.
    # Uncertainty product squared is (int x^2 |g|^2 / int |g|^2) * (int |g'|^2 / int |g|^2)
    # Note: Depending on FT convention, there might be factors of 2*pi. 
    # For Heisenberg, dx dxi >= 1/2. The product of standard deviations is what we minimize.
    
    g_prime_sq_norm = np.trapz(g_prime**2, x)
    
    var_x = x2_moment / norm_sq
    var_xi = g_prime_sq_norm / norm_sq
    
    # The uncertainty product is sqrt(var_x * var_xi)
    # We minimize this.
    return np.sqrt(var_x * var_xi)

def construct():
    # Initial guess from current program
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Run optimization
    try:
        # Use L-BFGS-B for unconstrained optimization
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return x0
            
        if res.success:
            return res.x
        else:
            return x0
    except Exception:
        return x0
