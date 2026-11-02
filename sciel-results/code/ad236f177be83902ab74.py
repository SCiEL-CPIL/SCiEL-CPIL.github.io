"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize
import time

def _compute_upper_bound(coeffs):
    """
    Compute the upper bound on C4 = A(f) * A(F) for the given Hermite coefficients.
    coeffs = [c0, c4, c8] for H0, H4, H8.
    The H12 coefficient is determined by g(0) = 0.
    """
    c0, c4, c8 = coeffs
    
    # Hermite polynomials: H_0(x)=1, H_4(x)=16x^4-48x^2+12, H_8(x)=256x^8-1344x^6+2688x^4-1344x^2+168
    # g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # g(0) = c0 + c4*12 + c8*168 + c12*4608 = 0
    # H12(0) = 4608 (physicist's Hermite polynomial)
    c12 = -(c0 + c4 * 12 + c8 * 168) / 4608.0
    
    # We need to compute A(f) and A(F) where f(x) = g(x) * exp(-x^2/2) or similar Gaussian.
    # The exact form depends on the notebook's definition. 
    # Typically for uncertainty principle bounds, we consider:
    # A(f) = sup_x |f(x)| * sqrt(pi) * something, or related to L-infinity norm.
    # Actually, for the uncertainty inequality A(f)A(F) >= C, 
    # A(f) is often defined as sup_x |f(x)| * (2*pi)^(-1/4) or similar.
    # Let's assume the standard setup where f(x) = sum c_k H_k(x) e^{-x^2/2}
    # and A(f) = sup_x |f(x)|, A(F) = sup_x |F(f)(x)|
    
    # To be efficient, we evaluate on a grid.
    x = np.linspace(-10, 10, 10001)
    exp_term = np.exp(-x**2 / 2)
    
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 1344*x**6 + 2688*x**4 - 1344*x**2 + 168
    # H12(x) = 4096x^12 - 46080x^10 + 241920x^8 - 580608x^6 + 645120x^4 - 302400x^2 + 4608
    H12 = 4096*x**12 - 46080*x**10 + 241920*x**8 - 580608*x**6 + 645120*x**4 - 302400*x**2 + 4608
    
    g = c0*H0 + c4*H4 + c8*H8 + c12*H12
    f = g * exp_term
    
    A_f = np.max(np.abs(f))
    
    # Fourier transform of f. Since f is a combination of Hermite-Gaussians,
    # its Fourier transform is also a combination of Hermite-Gaussians.
    # F(H_k(x) e^{-x^2/2}) = i^k H_k(x) e^{-x^2/2} (with appropriate normalization)
    # Using the convention F(f)(xi) = integral f(x) e^{-2*pi*i*x*xi} dx
    # Or the unitary convention. Let's assume the bound is invariant under scaling.
    # For the unitary Fourier transform, F(H_k(x) e^{-x^2/2}) = i^k H_k(x) e^{-x^2/2}
    # So A_F = max |sum c_k i^k H_k(x) e^{-x^2/2}|
    
    F0 = H0 * exp_term
    F4 = (1j)**4 * H4 * exp_term  # i^4 = 1
    F8 = (1j)**8 * H8 * exp_term  # i^8 = 1
    F12 = (1j)**12 * H12 * exp_term  # i^12 = 1
    
    F = c0*F0 + c4*F4 + c8*F8 + c12*F12
    A_F = np.max(np.abs(F))
    
    return A_f * A_F

def construct():
    deadline = time.time() + 180  # 180 second limit
    
    # Initial guess from Goncalves et al. 2017
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_score = _compute_upper_bound(x0)
    best_coeffs = x0.copy()
    
    # Try local optimization
    try:
        result = minimize(_compute_upper_bound, x0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if result.success or result.fun < best_score:
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        return best_coeffs
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation around the best solution
        x_try = best_coeffs + np.random.normal(0, 0.1, 3)
        
        try:
            result = minimize(_compute_upper_bound, x_try, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
