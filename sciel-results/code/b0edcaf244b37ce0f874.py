import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

def construct():
    """
    Minimize the upper bound C4 on A(f)*A(F(f)) for a Hermite-Gaussian test function.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by g(0) = 0.
    
    Returns array of three coefficients [a0, a4, a8].
    """
    
    # Pre-compute grid and Hermite polynomials for efficiency
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # Gaussian envelope
    exp_term = np.exp(-x**2 / 2)
    
    def objective(coeffs):
        """Compute A(f) * A(F(f)) where A is the integral of |g| (L1 norm)."""
        a0, a4, a8 = coeffs
        
        # Derive a12 from g(0) = 0
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        
        # Compute g(x)
        poly_x = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        g_x = poly_x * exp_term
        A_f = np.trapz(np.abs(g_x), x)
        
        # Compute Fourier transform g_hat(xi)
        # F(Hn(x)e^{-x^2/2}) = (i)^n Hn(xi) e^{-xi^2/2}
        # For n=0,4,8,12, (i)^n is 1,1,1,-1 respectively.
        # Note: The grid for xi is the same as x due to symmetry of the problem structure
        # and the fact that we are just integrating over the same domain.
        
        poly_f = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12  # (i)^12 = -1
        g_f = poly_f * exp_term
        A_Ff = np.trapz(np.abs(g_f), x)
        
        return A_f * A_Ff

    # Initial guess: the prior state of the art
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple starting points to avoid local minima
    best_result = None
    best_val = np.inf
    
    # Starting points to try
    starts = [
        x0,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-0.8, 0.05, 0.0002]),
        np.array([-1.5, 0.02, 0.0001]),
        np.array([-0.5, 0.06, 0.0004]),
        # Add some random perturbations around x0
        x0 + np.array([0.1, 0.01, 0.0001]),
        x0 + np.array([-0.1, -0.01, -0.0001]),
    ]
    
    for start in starts:
        try:
            # Use L-BFGS-B for better convergence on smooth parts
            # eps=1e-8 for numerical gradient
            result = minimize(objective, start, method='L-BFGS-B', 
                            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is not None:
        return best_result.x
    else:
        return x0
