import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    timeout = 180.0  # Safety margin before 210s limit
    
    # Hermite polynomials H_0, H_4, H_8, H_12
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 221760x^2 + 10395
    
    # Grid for numerical integration
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    w = np.exp(-x**2)
    
    # Precompute Hermite polynomials on grid
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = x**12 - 84*x**10 + 2520*x**8 - 30240*x**6 + 151200*x**4 - 221760*x**2 + 10395
    
    # Values at x=0 for constraint
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        # Calculate c12 such that P(0) = 0
        # c0*1 + c4*3 + c8*105 + c12*10395 = 0
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        
        # Construct P(x)
        P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
        
        # Calculate integrals
        # Numerator: int x^2 e^-x^2 P(x)^2 dx
        # Denominator: int e^-x^2 P(x)^2 dx
        P2 = P**2
        denom = np.trapz(w * P2, x)
        if denom < 1e-12:
            return 1e12
        num = np.trapz(x**2 * w * P2, x)
        
        # Variance sigma_x^2 = num/denom - mean_x^2
        # Since P is even (H_even), mean_x = 0.
        # So sigma_x^2 = num/denom
        return num / denom

    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Optimization
    best_result = None
    try:
        # Use Nelder-Mead
        # maxiter is a soft limit, we also check time
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        best_result = res.x
    except Exception:
        best_result = x0

    # Check if we have time for a second pass or refinement?
    # For now, return the best found.
    return best_result
