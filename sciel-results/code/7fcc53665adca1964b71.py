import numpy as np
from scipy.optimize import differential_evolution
import sympy as sp

# Pre-compute exact Hermite polynomial coefficients as rational numbers
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 65536x^8 - 147456x^6 + 116128x^4 - 23040x^2 + 1344
# H12(x) = 16777216x^12 - 125829120x^10 + 331776000x^8 - 331776000x^6 + 119439360x^4 - 14745600x^2 + 479001

# Values at x=0
H0_0 = sp.Rational(1)
H4_0 = sp.Rational(12)
H8_0 = sp.Rational(1344)
H12_0 = sp.Rational(479001)

# Coefficients of x^(2k) in each Hermite polynomial (k=0 to 6)
# H0: [1]
# H4: [12, -48, 16] for x^0, x^2, x^4
# H8: [1344, -23040, 116128, -147456, 65536] for x^0, x^2, x^4, x^6, x^8
# H12: [479001, -14745600, 119439360, -331776000, 331776000, -125829120, 16777216] for x^0, x^2, ..., x^12

H0_even = [sp.Rational(1)]
H4_even = [sp.Rational(12), sp.Rational(-48), sp.Rational(16)]
H8_even = [sp.Rational(1344), sp.Rational(-23040), sp.Rational(116128), sp.Rational(-147456), sp.Rational(65536)]
H12_even = [sp.Rational(479001), sp.Rational(-14745600), sp.Rational(119439360), sp.Rational(-331776000), sp.Rational(331776000), sp.Rational(-125829120), sp.Rational(16777216)]

def construct():
    # Seed coefficients
    c0_seed, c1_seed, c2_seed = -113 / 100, 1 / 25, 1 / 3240
    
    def compute_Q_coeffs(c0, c1, c2):
        """Compute coefficients of Q(y) = g(sqrt(y))/y using exact rational arithmetic."""
        c0_r = sp.Rational(c0)
        c1_r = sp.Rational(c1)
        c2_r = sp.Rational(c2)
        
        # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
        c3_r = -(c0_r * H0_0 + c1_r * H4_0 + c2_r * H8_0) / H12_0
        
        # A[k] = coefficient of x^(2k) in g(x)
        A = [sp.Rational(0)] * 7
        for k in range(len(H0_even)):
            A[k] += c0_r * H0_even[k]
        for k in range(len(H4_even)):
            A[k] += c1_r * H4_even[k]
        for k in range(len(H8_even)):
            A[k] += c2_r * H8_even[k]
        for k in range(len(H12_even)):
            A[k] += c3_r * H12_even[k]
        
        # Q(y) = A[1] + A[2]*y + A[3]*y^2 + A[4]*y^3 + A[5]*y^4 + A[6]*y^5
        # Convert to float array for np.roots
        Q_coeffs = [float(A[1]), float(A[2]), float(A[3]), float(A[4]), float(A[5]), float(A[6])]
        return Q_coeffs
    
    def objective(params):
        c0, c1, c2 = params
        
        Q_coeffs = compute_Q_coeffs(c0, c1, c2)
        
        # Find roots of Q(y)
        roots = np.roots(Q_coeffs)
        
        # Filter real positive roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-12 and r.real > 1e-12:
                real_roots.append(r.real)
                
        if not real_roots:
            return 1e10
            
        # Sort roots
        real_roots.sort()
        
        # Find largest root with sign change
        max_y = 0
        for y_r in reversed(real_roots):
            # Check sign change
            delta = 1e-8 * max(1.0, y_r)
            y_minus = y_r - delta
            y_plus = y_r + delta
            
            if y_minus < 1e-12:
                continue
                
            q_minus = np.polyval(Q_coeffs, y_minus)
            q_plus = np.polyval(Q_coeffs, y_plus)
            
            if q_minus * q_plus < 0:
                max_y = y_r
                break
                
        if max_y == 0:
            return 1e10
            
        return max_y
    
    # Use differential evolution for global optimization
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=200,
        popsize=15,
        tol=1e-10,
        atol=1e-10,
        polish=True
    )
    
    # If optimization fails or returns worse, fall back to seed
    seed_value = objective(np.array([c0_seed, c1_seed, c2_seed]))
    if result.fun > seed_value:
        return np.array([c0_seed, c1_seed, c2_seed])
    
    return result.x
