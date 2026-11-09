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
    
    # Precompute H_n(0) values
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1344.0
    H12_0 = -103216.0
    
    # Precompute integration grid
    x_grid = np.linspace(-12, 12, 8001)
    
    # Precompute Hermite polynomials on the grid
    H0_grid = eval_hermite(0, x_grid)
    H4_grid = eval_hermite(4, x_grid)
    H8_grid = eval_hermite(8, x_grid)
    H12_grid = eval_hermite(12, x_grid)
    
    # Precompute Gaussian envelope
    gauss = np.exp(-x_grid**2 / 2)
    
    # Precompute weighted basis functions for g(x)
    basis_g = np.column_stack([H0_grid * gauss, H4_grid * gauss, H8_grid * gauss, H12_grid * gauss])
    
    # For Fourier transform: F(Hn * e^{-x^2/2}) = (i)^n * Hn(xi) * e^{-xi^2/2}
    # (i)^0 = 1, (i)^4 = 1, (i)^8 = 1, (i)^12 = -1
    basis_f = np.column_stack([H0_grid * gauss, H4_grid * gauss, H8_grid * gauss, -H12_grid * gauss])
    
    def get_a12(a0, a4, a8):
        """Compute a12 from g(0) = 0 constraint."""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def objective(coeffs):
        """Compute A(f) * A(F(f)) where A is the integral of |g| (L1 norm)."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        # Compute g(x) and F(g)(xi) using precomputed basis
        g_x = a0 * basis_g[:, 0] + a4 * basis_g[:, 1] + a8 * basis_g[:, 2] + a12 * basis_g[:, 3]
        g_f = a0 * basis_f[:, 0] + a4 * basis_f[:, 1] + a8 * basis_f[:, 2] + a12 * basis_f[:, 3]
        
        A_f = np.trapz(np.abs(g_x), x_grid)
        A_Ff = np.trapz(np.abs(g_f), x_grid)
        
        return A_f * A_Ff
    
    # Initial guess: the prior state of the art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Multiple starting points for robustness
    starts = [
        x0,
        np.array([-1.1, 0.04, 0.0003]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-0.9, 0.05, 0.0002]),
        np.array([-1.3, 0.025, 0.0004]),
        np.array([-0.7, 0.06, 0.0003]),
        np.array([-1.0, 0.045, 0.00035]),
        np.array([-1.15, 0.035, 0.00045]),
    ]
    
    best_val = np.inf
    best_coeffs = x0.copy()
    
    # Try Nelder-Mead with tight tolerances
    for start in starts:
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Refine with BFGS from the best Nelder-Mead result
    try:
        result2 = minimize(objective, best_coeffs, method='BFGS',
                          options={'maxiter': 2000, 'gtol': 1e-10, 'disp': False})
        if result2.fun < best_val:
            best_val = result2.fun
            best_coeffs = result2.x.copy()
    except Exception:
        pass
    
    # Further refine with L-BFGS-B
    try:
        result3 = minimize(objective, best_coeffs, method='L-BFGS-B',
                          options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14, 'disp': False})
        if result3.fun < best_val:
            best_val = result3.fun
            best_coeffs = result3.x.copy()
    except Exception:
        pass
    
    return best_coeffs
