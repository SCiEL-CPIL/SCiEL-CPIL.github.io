import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 150  # Safety margin for 180s limit
    
    # Factorials needed
    # k! for k in 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13
    fact = [1.0] * 14
    for i in range(1, 14):
        fact[i] = fact[i-1] * i
        
    # Hermite polynomials at 0: H_0=1, H_4=3, H_8=15, H_12=105
    # H_n(0) for even n is (-1)^{n/2} (n-1)!!
    # H_0(0) = 1
    # H_4(0) = 3
    # H_8(0) = 15
    # H_12(0) = 105
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 15.0*c8) / 105.0

    def objective(c):
        c0, c4, c8 = c
        c12 = get_c12(c0, c4, c8)
        
        # Norm squared N = sqrt(pi) * sum(c_k^2 * k!)
        # We can ignore sqrt(pi) as it cancels in the ratio
        N = c0**2 * fact[0] + c4**2 * fact[4] + c8**2 * fact[8] + c12**2 * fact[12]
        
        # If N is too small, return large value to avoid division by zero
        if N < 1e-10:
            return 1e10
            
        # Derivative norm squared D = sqrt(pi) * sum(c_k^2 * k * k! for k in 4,8,12)
        # Term for k: c_k^2 * k * k!
        D = c4**2 * 4 * fact[4] + c8**2 * 8 * fact[8] + c12**2 * 12 * fact[12]
        
        # x*g norm squared X = sqrt(pi) * sum(coeff_j^2 * j!)
        # Coefficients for H_j in x*g:
        # j=1: c0/2
        # j=3: 2*c4
        # j=5: c4/2
        # j=7: 4*c8
        # j=9: c8/2
        # j=11: 6*c12
        # j=13: c12/2
        
        X = (c0/2.0)**2 * fact[1] + \
            (2.0*c4)**2 * fact[3] + \
            (c4/2.0)**2 * fact[5] + \
            (4.0*c8)**2 * fact[7] + \
            (c8/2.0)**2 * fact[9] + \
            (6.0*c12)**2 * fact[11] + \
            (c12/2.0)**2 * fact[13]
            
        # Objective: (X * D) / (N * N)
        return (X * D) / (N * N)

    # Initial guess from seed program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try to optimize
    # Using L-BFGS-B as it handles bounds well (though we don't have explicit bounds, 
    # the function is well-behaved near the seed)
    
    best_x = x0
    best_val = objective(x0)
    
    # Multi-start to be safe, but keep it fast
    # We can perturb the seed slightly
    starts = [
        x0,
        x0 * 1.1,
        x0 * 0.9,
        x0 + np.array([0.1, 0.01, 0.001]),
        x0 - np.array([0.1, 0.01, 0.001])
    ]
    
    for start in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, start, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return np.array(best_x)
