import numpy as np
from scipy.optimize import minimize, differential_evolution
from sympy.polys.orthopolys import hermite_poly
import sympy as sp

# Precompute Hermite polynomials coefficients
# H_n(x) are physicists' Hermite polynomials.
# We need H0, H4, H8, H12.

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

H0_padded = np.array(pad_coeffs(H0_coeffs, 12))
H4_padded = np.array(pad_coeffs(H4_coeffs, 12))
H8_padded = np.array(pad_coeffs(H8_coeffs, 12))
H12_padded = np.array(pad_coeffs(H12_coeffs, 12))

# Precompute values at 0
H0_0 = H0_padded[-1]
H4_0 = H4_padded[-1]
H8_0 = H8_padded[-1]
H12_0 = H12_padded[-1]

def construct():
    # Initial seed
    c0_seed, c1_seed, c2_seed = -113 / 100, 1 / 25, 1 / 3240
    
    def objective(coeffs):
        c0, c1, c2 = coeffs
        
        # Compute c3
        val_0 = c0 * H0_0 + c1 * H4_0 + c2 * H8_0
        c3 = -val_0 / H12_0
        
        # Construct g coefficients
        g_coeffs = c0 * H0_padded + c1 * H4_padded + c2 * H8_padded + c3 * H12_padded
        
        # Check leading coefficient sign
        # If leading coeff is negative, g(x) -> -infinity. Evaluator flips g.
        # Flipping g changes roots? No, roots are same.
        # But "sign change" direction flips.
        # However, the set of roots is identical.
        # The value depends on the largest root where sign changes.
        # If we flip g, the sign change at r is reversed, but it still changes sign.
        # So the set of candidate roots is the same.
        # We just need to ensure we handle the "non-negativity" check correctly if we assume we want g >= 0.
        # If we want to find a polynomial that is >= 0, the sign of the leading coeff matters for the "infinity" behavior.
        # If lead < 0, g -> -inf. It cannot be >= 0 for all x.
        # So we require lead > 0 for the "non-negative" strategy to work.
        
        lead_coeff = g_coeffs[0]
        if lead_coeff <= 0:
            # If we can't be non-negative, we still want to minimize the largest root.
            # But if we are targeting non-negativity, this is a hard fail.
            # Let's just return a penalty to steer away, or handle it.
            # Actually, if lead < 0, the evaluator flips g. Then lead > 0.
            # So effectively, we can assume lead > 0 by taking abs or flipping.
            # Let's just flip if lead < 0 for our internal calculation of "non-negativity".
            g_coeffs = -g_coeffs
            # c3 effectively changed sign, but roots are same.
        
        # Find roots
        try:
            roots = np.roots(g_coeffs)
        except Exception:
            return 1e10
            
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        positive_roots = real_roots[real_roots > 1e-9]
        
        if len(positive_roots) == 0:
             # No positive roots.
             # Check if g(x) >= 0 for all x >= 0.
             # Sample grid
             x_max = 10.0
             x_grid = np.linspace(0, x_max, 1000)
             g_vals = np.polyval(g_coeffs, x_grid)
             if np.any(g_vals < -1e-9):
                 # It went negative but came back? Or just negative somewhere?
                 # If it went negative, there must be roots.
                 # If no positive roots found, maybe roots are complex?
                 # If g(x) < 0 somewhere and g(0)=0 and g(inf)=inf, it must cross.
                 # So if no positive roots, it's likely always positive (or zero).
                 # Let's assume it's a valid non-negative polynomial.
                 return 1e-6 # Very small value, high score
             else:
                 # Always positive
                 return 1e-6

        # Check for sign changes
        # We want the largest root where sign changes.
        # Sort positive roots
        positive_roots.sort()
        
        # Check sign at 0+ and at infinity (positive)
        # g(0) = 0.
        # Derivative at 0?
        # Let's just check sign at midpoints between roots.
        
        # If there are no sign changes, return 1e-6?
        # If g(x) >= 0, roots are even multiplicity. No sign change.
        # The problem says "r is the largest real root ... at which q changes sign".
        # If no such root, what is value?
        # If we assume 0 is the best value, we return 0.
        
        r_max_sign_change = 0
        prev_sign = 1 # Assume g(x) > 0 for small x > 0?
        # Check sign at small epsilon
        eps = 1e-5
        g_eps = np.polyval(g_coeffs, eps)
        if g_eps < 0:
            prev_sign = -1
        else:
            prev_sign = 1
            
        for i in range(len(positive_roots)):
            r = positive_roots[i]
            # Check sign after r
            next_x = r + 1e-5
            # If r is the last root, check sign at r + 1e-5
            # But we need to know if it changes sign AT r.
            # If multiplicity is odd, it changes.
            # We can check sign at r - delta and r + delta.
            g_before = np.polyval(g_coeffs, max(eps, r - 1e-5))
            g_after = np.polyval(g_coeffs, r + 1e-5)
            
            sign_before = 1 if g_before > 0 else -1
            sign_after = 1 if g_after > 0 else -1
            
            if sign_before != sign_after:
                r_max_sign_change = r
            prev_sign = sign_after
        
        if r_max_sign_change > 0:
            return r_max_sign_change ** 2
        else:
            # No sign changing roots.
            # Check if g(x) >= 0
            x_max = max(10.0, 2.0 * (positive_roots[-1] if len(positive_roots)>0 else 1.0))
            x_grid = np.linspace(0, x_max, 1000)
            g_vals = np.polyval(g_coeffs, x_grid)
            if np.any(g_vals < -1e-9):
                # It dipped negative but no sign change detected?
                # Maybe numerical error in root finding or sign check.
                # Return a large penalty to avoid this state if it's invalid.
                # Or maybe it's a valid state with value 0?
                # If g(x) < 0, it's not non-negative.
                # If the evaluator requires non-negativity, this is bad.
                # If the evaluator just takes the largest sign-changing root, and there are none,
                # but g is negative somewhere, it's a contradiction (must cross).
                # So likely numerical issue.
                return 1e6
            else:
                return 1e-6

    # Use differential evolution for global search
    bounds = [(-10, 10), (-10, 10), (-1, 1)]
    # Seed is [-1.13, 0.04, 0.0003]
    # Let's try to find a better configuration.
    
    # Run optimization
    # differential_evolution is robust but slow.
    # Let's try Nelder-Mead from multiple starting points including the seed.
    
    best_x = np.array([c0_seed, c1_seed, c2_seed])
    best_val = objective(best_x)
    
    # Try a few random perturbations
    np.random.seed(42)
    for i in range(20):
        x0 = np.array([c0_seed, c1_seed, c2_seed]) + np.random.normal(0, 0.1, 3)
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
            
    return best_x
