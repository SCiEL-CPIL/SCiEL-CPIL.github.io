import numpy as np
from scipy.optimize import differential_evolution

# Hardcoded coefficients for Physicists' Hermite Polynomials H_n(x)
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 65536x^8 - 147456x^6 + 116128x^4 - 23040x^2 + 1344
# H_12(x) = 16777216x^12 - 125829120x^10 + 331776000x^8 - 331776000x^6 + 119439360x^4 - 14745600x^2 + 479001

# Coefficients for powers x^0, x^2, x^4, x^6, x^8, x^10, x^12
# Index 0: x^0, Index 1: x^2, ..., Index 6: x^12
H0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H4 = np.array([12.0, -48.0, 16.0, 0.0, 0.0, 0.0, 0.0])
H8 = np.array([1344.0, -23040.0, 116128.0, -147456.0, 65536.0, 0.0, 0.0])
H12 = np.array([479001.0, -14745600.0, 119439360.0, -331776000.0, 331776000.0, -125829120.0, 16777216.0])

# Values at x=0
H0_0 = H0[0]
H4_0 = H4[0]
H8_0 = H8[0]
H12_0 = H12[0]

def construct():
    # Seed coefficients
    c0_seed, c1_seed, c2_seed = -113 / 100, 1 / 25, 1 / 3240
    
    def objective(params):
        c0, c1, c2 = params
        
        # Calculate c3 such that g(0) = 0
        # g(0) = c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
        c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
        
        # Construct g(x) coefficients A_k for x^(2k)
        # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
        A = c0 * H0 + c1 * H4 + c2 * H8 + c3 * H12
        
        # q(x) = g(x) / x^2
        # If g(x) = A0 + A1*x^2 + A2*x^4 + ... + A6*x^12
        # q(x) = A0/x^2 + A1 + A2*x^2 + ...
        # The task states "Let q = g / x^2, which must divide exactly".
        # This implies g(0) = 0, which is guaranteed by our c3 calculation.
        # So A0 should be 0 (or very close).
        # q(x) is a polynomial of degree 10 (in x) or degree 5 (in y=x^2).
        # q(x) = A1 + A2*x^2 + A3*x^4 + A4*x^6 + A5*x^8 + A6*x^10
        # Let y = x^2. Q(y) = A1 + A2*y + A3*y^2 + A4*y^3 + A5*y^4 + A6*y^5
        
        Q_coeffs = A[1:7] # [A1, A2, A3, A4, A5, A6]
        
        # Check if Q is effectively zero or invalid
        if np.all(np.abs(Q_coeffs) < 1e-12):
            return 1e10
            
        # Find roots of Q(y) = 0
        # np.roots expects coefficients from highest degree to lowest
        # Q(y) = A6*y^5 + A5*y^4 + A4*y^3 + A3*y^2 + A2*y + A1
        Q_rev = Q_coeffs[::-1]
        
        try:
            roots = np.roots(Q_rev)
        except Exception:
            return 1e10
            
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-9 and r.real > 1e-9:
                real_roots.append(r.real)
                
        if not real_roots:
            return 1e10
            
        real_roots.sort()
        
        # Find the largest root where sign changes
        # We check from largest to smallest
        max_y = 0
        for y_r in reversed(real_roots):
            delta = 1e-8 * max(1.0, y_r)
            y_minus = y_r - delta
            y_plus = y_r + delta
            
            if y_minus < 1e-12:
                continue
                
            q_minus = np.polyval(Q_rev, y_minus)
            q_plus = np.polyval(Q_rev, y_plus)
            
            if q_minus * q_plus < 0:
                max_y = y_r
                break
                
        if max_y == 0:
            return 1e10
            
        # value = r^2 / (2 * pi)
        # r = sqrt(y_r)
        # value = y_r / (2 * pi)
        return max_y / (2 * np.pi)

    # Define bounds for Differential Evolution
    # Seed: c0 ~ -1, c1 ~ 0.04, c2 ~ 0.0003
    # We can try a wider range to find a global minimum
    bounds = [
        (-5.0, 5.0),    # c0
        (-1.0, 1.0),    # c1
        (-0.01, 0.01)   # c2
    ]
    
    # Use Differential Evolution for global search
    # This is significantly more robust than Nelder-Mead for this non-smooth objective
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=500, 
        tol=1e-8,
        polish=False # Disable Nelder-Mead polish to avoid slow local opt
    )
    
    best_params = result.x
    best_val = result.fun
    
    # Compare with seed
    seed_val = objective([c0_seed, c1_seed, c2_seed])
    
    if best_val < seed_val:
        return best_params
    else:
        return np.array([c0_seed, c1_seed, c2_seed])
