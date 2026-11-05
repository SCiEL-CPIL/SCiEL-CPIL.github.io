import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre, factorial

def _compute_uncertainty_fast(roots):
    roots = np.array(roots)
    p_coeffs = np.poly(roots)
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)
    
    def moment(k):
        total = 0.0
        for j in range(25):
            coeff = p2_coeffs[24-j]
            if coeff != 0:
                total += coeff * factorial(k+j)
        return total
    
    m0 = moment(0)
    m1 = moment(1)
    m2 = moment(2)
    
    if m0 == 0:
        return 1e10
        
    mean_x = m1 / m0
    var_x = m2 / m0 - mean_x**2
    
    # Precompute factorials
    facts = np.array([factorial(j) for j in range(25)])
    coeffs = p2_coeffs[24::-1] # c_0 to c_24
    
    def W(xi):
        denom = 1.0 + 1j*xi
        # vectorized power calculation might be tricky with negative powers
        # loop is fine for 25 terms
        total = np.zeros_like(xi, dtype=complex)
        for j in range(25):
            if coeffs[j] != 0:
                total += coeffs[j] * facts[j] * (denom ** (-(j+1)))
        return total
        
    xi_max = 20.0
    num_points = 2000
    xi_grid = np.linspace(-xi_max, xi_max, num_points)
    
    w_vals = W(xi_grid)
    
    denom_int = 2.0 * np.pi * m0
    num_xi = np.trapz(xi_grid * w_vals, xi_grid)
    num_xi2 = np.trapz(xi_grid**2 * w_vals, xi_grid)
    
    mean_xi = np.real(num_xi) / denom_int
    var_xi = np.real(num_xi2) / denom_int - mean_xi**2
    
    if var_x < 0 or var_xi < 0:
        return 1e10
        
    return np.sqrt(var_x * var_xi)

def construct():
    roots, _ = roots_laguerre(12)
    initial_roots = np.array(roots)
    
    def objective(x):
        if np.any(x <= 0):
            return 1e10
        try:
            return _compute_uncertainty_fast(x)
        except Exception:
            return 1e10

    res = minimize(objective, initial_roots, method='Nelder-Mead', 
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
    
    if res.fun < _compute_uncertainty_fast(initial_roots):
        final_roots = res.x
    else:
        final_roots = initial_roots
        
    final_roots = np.sort(np.abs(final_roots))
    
    return [float(r) for r in final_roots]
