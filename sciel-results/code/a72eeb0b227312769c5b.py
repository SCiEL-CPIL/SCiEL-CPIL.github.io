import numpy as np
from scipy.optimize import differential_evolution

def construct():
    # Coefficients of Hermite polynomials (Probabilists' He_n) in descending power order
    # He_0(x) = 1
    # He_4(x) = x^4 - 6x^2 + 3
    # He_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # He_12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 10395
    
    # We work with polynomials of degree 12.
    # Represented as arrays of length 13, index i corresponds to x^(12-i)
    
    # He12: x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 10395
    he12_coeffs = np.array([1, 0, -66, 0, 1155, 0, -6930, 0, 13860, 0, -7560, 0, 10395])
    
    # He8: x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # Pad to length 13
    he8_coeffs = np.zeros(13)
    he8_coeffs[4:] = [1, 0, -28, 0, 210, 0, -420, 0, 105] # Degrees 8 down to 0
    
    # He4: x^4 - 6x^2 + 3
    he4_coeffs = np.zeros(13)
    he4_coeffs[8:] = [1, 0, -6, 0, 3] # Degrees 4 down to 0
    
    # He0: 1
    he0_coeffs = np.zeros(13)
    he0_coeffs[12] = 1.0 # Degree 0
    
    def get_P_coeffs(c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        # P(x) = c0*He0 + c4*He4 + c8*He8 + c12*He12
        return c0*he0_coeffs + c4*he4_coeffs + c8*he8_coeffs + c12*he12_coeffs

    def objective(c):
        c0, c4, c8 = c
        P_coeffs = get_P_coeffs(c0, c4, c8)
        
        # P(x) is degree 12.
        # P'(x) is degree 11.
        # x*P(x) is degree 13.
        # Q(x) = P'(x) - x*P(x) is degree 13.
        
        # Derivative coefficients
        # If P_coeffs[i] is coeff of x^(12-i), then P'(x) coeff of x^(11-i) is (12-i)*P_coeffs[i]
        # P' has degree 11. Array length 12.
        dP_coeffs = np.zeros(12)
        for i in range(12):
            dP_coeffs[i] = (12 - i) * P_coeffs[i]
            
        # x*P(x) coefficients
        # x*P(x) has degree 13. Array length 14.
        # Coeff of x^13 is 0.
        # Coeff of x^12 is P_coeffs[0] (coeff of x^12 in P)
        # ...
        xP_coeffs = np.zeros(14)
        xP_coeffs[1:] = P_coeffs
        
        # Q(x) = P'(x) - x*P(x)
        # Pad dP_coeffs to length 14
        dP_padded = np.zeros(14)
        dP_padded[2:] = dP_coeffs # dP has deg 11, so it starts at x^11, which is index 2 in a deg 13 poly (x^13, x^12, x^11...)
        
        Q_coeffs = dP_padded - xP_coeffs
        
        # Find roots of Q(x)
        roots = np.roots(Q_coeffs)
        
        # Filter real roots
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        
        # Evaluate |h(x)| = |P(x) * exp(-x^2/2)| at real roots
        max_val = 0.0
        if len(real_roots) > 0:
            # Evaluate P(x) at roots
            P_vals = np.polyval(P_coeffs, real_roots)
            h_vals = P_vals * np.exp(-real_roots**2 / 2)
            max_val = np.max(np.abs(h_vals))
            
        # Also check x=0 just in case (though it should be a root if P'(0)=0)
        # And maybe check a few large x? No, exp decays.
        # If no roots found (unlikely for odd degree poly), return a large value
        if len(real_roots) == 0:
             return 1e6
             
        return max_val

    # Use Differential Evolution
    # Bounds for c0, c4, c8
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    # Run DE
    # seed for reproducibility
    res = differential_evolution(objective, bounds, seed=42, maxiter=200, popsize=20, tol=1e-8, polish=True)
    
    return res.x
