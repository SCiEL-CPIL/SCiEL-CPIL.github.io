"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin
    
    # Hermite polynomials H0, H4, H8, H12
    # We need to compute c12 from c0, c4, c8 such that g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 9240
    
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 9240.0
    
    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    def objective(coeffs):
        if time.time() > deadline:
            return 1e10
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Grid for numerical integration
        # Range [-10, 10] is sufficient for Gaussian
        x = np.linspace(-10, 10, 1000)
        
        # Compute g(x)
        H0 = eval_hermite(0, x)
        H4 = eval_hermite(4, x)
        H8 = eval_hermite(8, x)
        H12 = eval_hermite(12, x)
        
        g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-x**2 / 2)
        
        # Compute Fourier Transform g_hat(xi)
        # FT of Hn(x)exp(-x^2/2) is (-i)^n Hn(xi)exp(-xi^2/2)
        # Assuming standard unitary or non-unitary? 
        # Usually A(f)A(fhat) is invariant to scaling. 
        # Let's use the property that the shape is the same.
        xi = x # Symmetric grid
        H0_hat = eval_hermite(0, xi)
        H4_hat = eval_hermite(4, xi)
        H8_hat = eval_hermite(8, xi)
        H12_hat = eval_hermite(12, xi)
        
        # Phase factors: (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=1
        # Wait, (-i)^2 = -1. (-i)^4 = 1. 
        # So all even n have phase 1.
        # This means g_hat(xi) has the same functional form as g(x) if coefficients are real.
        # g_hat(xi) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-xi^2/2)
        # So ||g||_1 == ||g_hat||_1.
        # If A(f) is proportional to ||f||_1, then we minimize ||g||_1^2.
        
        # However, if A(f) is defined differently (e.g. support width), this might not hold.
        # But for even functions, the L1 norm of g and g_hat are identical.
        # So minimizing the L1 norm of g is a good proxy.
        
        # To avoid trivial zero solution, we might need a constraint or regularization.
        # The seed values are non-zero. 
        # If we just minimize L1 norm, we can scale down to 0.
        # The task implies a fixed "class" or normalization.
        # "Tune a ... test function". Usually implies normalized L2 norm = 1.
        
        # Let's enforce L2 norm = 1.
        # ||g||_2^2 = integral |g|^2 dx
        # Use np.trapz
        norm_sq = np.trapz(np.abs(g)**2, x)
        
        if norm_sq < 1e-12:
            return 1e10
            
        # Normalize g to L2 norm 1
        g_norm = g / np.sqrt(norm_sq)
        
        # Calculate L1 norm
        l1_norm = np.trapz(np.abs(g_norm), x)
        
        # The uncertainty product is often related to l1(g) * l1(g_hat)
        # Since l1(g) == l1(g_hat), we minimize l1(g)^2
        # Or just l1(g)
        
        return l1_norm

    # Initial guess from seed
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    # Optimize
    try:
        res = minimize(objective, [c0_init, c4_init, c8_init], method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success:
            c0, c4, c8 = res.x
            c12 = get_c12(c0, c4, c8)
            return np.array([c0, c4, c8])
        else:
            # Fallback to seed if optimization fails
            return np.array([c0_init, c4_init, c8_init])
            
    except Exception:
        return np.array([c0_init, c4_init, c8_init])

# EVOLVE-BLOCK-END
