import numpy as np
import scipy.optimize as opt

def construct():
    # Weights B_i = 2^n * n!
    n_vals = [0, 4, 8, 12]
    B = [2**n * int(np.math.factorial(n)) for n in n_vals]
    # A_i = (n + 0.5) * B_i
    A = [(n + 0.5) * b for n, b in zip(n_vals, B)]
    
    # Constraint coefficients for g(0) = 0
    # H_0(0)=1, H_4(0)=12, H_8(0)=3024, H_12(0)=4790016
    h0, h4, h8, h12 = 1, 12, 3024, 4790016
    
    def objective(c):
        c0, c4, c8 = c
        c12 = -(c0 + h4*c4 + h8*c8) / h12
        cs = [c0, c4, c8, c12]
        
        # Calculate numerator and denominator
        num = sum(ci**2 * ai for ci, ai in zip(cs, A))
        den = sum(ci**2 * bi for ci, bi in zip(cs, B))
        
        if den == 0:
            return 1e10
        return num / den

    # Initial guess from seed program
    # Seed: -113/100, 1/25, 1/3240
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Use Nelder-Mead for robustness without gradient
    res = opt.minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000})
    
    c0, c4, c8 = res.x
    return np.array([c0, c4, c8])
