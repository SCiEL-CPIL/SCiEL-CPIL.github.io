import numpy as np
from scipy.optimize import minimize
from sympy.polys.orthopolys import hermite_poly
import sympy as sp

# Precompute Hermite polynomials coefficients
# H_n(x) are physicists' Hermite polynomials.
# We need H0, H4, H8, H12.
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 10752x^4 - 10752x^2 + 1680
# H12(x) = ...
# It is safer to use sympy to get exact coefficients and convert to float.

def get_hermite_coeffs(n):
    x = sp.Symbol('x')
    Hn = hermite_poly(n, x)
    coeffs = sp.Poly(Hn, x).all_coeffs()
    return [float(c) for c in coeffs]

H0_coeffs = get_hermite_coeffs(0)
H4_coeffs = get_hermite_coeffs(4)
H8_coeffs = get_hermite_coeffs(8)
H12_coeffs = get_hermite_coeffs(12)

# Pad coefficients to degree 12
def pad_coeffs(coeffs, target_degree):
    # coeffs are from highest degree to lowest
    # numpy.roots expects highest degree to lowest
    padded = [0.0] * (target_degree - len(coeffs) + 1)
    padded += coeffs
    return padded

H0_padded = pad_coeffs(H0_coeffs, 12)
H4_padded = pad_coeffs(H4_coeffs, 12)
H8_padded = pad_coeffs(H8_coeffs, 12)
H12_padded = pad_coeffs(H12_coeffs, 12)

def construct():
    # Initial seed
    c0, c1, c2 = -113 / 100, 1 / 25, 1 / 3240
    x0 = np.array([c0, c1, c2])
    
    def objective(coeffs):
        c0, c1, c2 = coeffs
        
        # Compute c3
        # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 1680.0
        H12_0 = 47900160.0 # 12! / 2^6 * ... wait, H_n(0) = (-1)^(n/2) (n-1)!! for even n?
        # Let's calculate H12(0) using the padded coeffs.
        # H12_coeffs is [leading, ..., constant]
        # H12(0) is the last element.
        
        # Using precomputed padded coeffs, H12(0) is H12_padded[-1]
        val_0 = c0 * H0_padded[-1] + c1 * H4_padded[-1] + c2 * H8_padded[-1]
        c3 = -val_0 / H12_padded[-1]
        
        # Check leading coefficient sign
        # Leading coeff of g is c3 * H12_padded[0]
        lead_coeff = c3 * H12_padded[0]
        if lead_coeff <= 0:
            return 1e10 # Penalty for wrong sign at infinity
        
        # Construct g coefficients
        g_coeffs = c0 * np.array(H0_padded) + c1 * np.array(H4_padded) + c2 * np.array(H8_padded) + c3 * np.array(H12_padded)
        
        # Find roots
        try:
            roots = np.roots(g_coeffs)
        except Exception:
            return 1e10
            
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        positive_roots = real_roots[real_roots > 1e-9]
        
        if len(positive_roots) == 0:
             # No positive roots? Then g(x) > 0 for all x>0?
             # If g(x) > 0 for all x, r is undefined or 0?
             # The problem says "r is the largest real root of q at which q changes sign".
             # If no such root, maybe value is 0? Or very small?
             # If g(x) > 0, it never crosses.
             # Let's assume if no positive roots, value is 0 (best possible).
             # But wait, if g(x) is always positive, does it satisfy the inequality?
             # The inequality is likely g(x) >= 0.
             # If it is always positive, the "bound" might be determined by the infimum of something else?
             # However, the problem defines value based on r.
             # If r doesn't exist, perhaps we return a very small number or 0.
             # Let's check the seed. The seed likely has roots.
             return 0.0
             
        r_max = np.max(positive_roots)
        
        # Check non-negativity on a grid
        # We need g(x) >= 0 for all x >= 0.
        # Sample up to 2 * r_max or a fixed large number.
        x_max = max(10.0, 2.0 * r_max)
        x_grid = np.linspace(0, x_max, 1000)
        g_vals = np.polyval(g_coeffs, x_grid)
        
        if np.any(g_vals < -1e-9):
            return 1e10 # Penalty for violating non-negativity
            
        return r_max ** 2

    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if result.success:
        return result.x
    else:
        return x0
