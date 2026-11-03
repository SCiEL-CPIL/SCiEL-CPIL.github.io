"""Rewritten program using global optimization (Differential Evolution) for the uncertainty inequality task."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    
    We minimize the product of the spatial variance and the frequency variance
    (which corresponds to the upper bound on the uncertainty constant C4).
    """
    # Coefficients for a12 in terms of a0, a4, a8
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 10395
    # a0 + 3a4 + 105a8 + 10395a12 = 0
    c0 = 1.0 / 10395.0
    c4 = 3.0 / 10395.0
    c8 = 105.0 / 10395.0
    
    # Factorials
    f0 = 1.0       # 0!
    f4 = 24.0      # 4!
    f8 = 40320.0   # 8!
    f12 = 479001600.0 # 12!
    
    # Coefficients for <x^2 g, g> diagonal terms: (2n+1) * n!
    x2_0 = 1.0
    x2_4 = 216.0
    x2_8 = 685440.0
    x2_12 = 11975040000.0
    
    # Coefficients for <g', g'> diagonal terms: n * n!
    k_0 = 0.0
    k_4 = 96.0
    k_8 = 322560.0
    k_12 = 5748019200.0

    def get_a12(a0, a4, a8):
        return -(c0 * a0 + c4 * a4 + c8 * a8)

    def objective(coeffs):
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        # Norm squared N
        N = a0**2 * f0 + a4**2 * f4 + a8**2 * f8 + a12**2 * f12
        
        if N < 1e-15:
            return 1e15
            
        # Spatial moment X
        X = a0**2 * x2_0 + a4**2 * x2_4 + a8**2 * x2_8 + a12**2 * x2_12
        
        # Frequency moment K
        K = a0**2 * k_0 + a4**2 * k_4 + a8**2 * k_8 + a12**2 * k_12
        
        val = (X * K) / (N * N)
        return val

    # Bounds for the coefficients
    # The problem is scale invariant, so we can bound the coefficients.
    # If coefficients are too large, numerical issues might arise, but double precision is fine.
    # We use a reasonably wide range to ensure global coverage.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use Differential Evolution for global optimization
    # seed is fixed for reproducibility
    res = differential_evolution(objective, bounds, seed=42, maxiter=1000, tol=1e-12, polish=True)
    
    if res.success and res.fun < 1e14:
        best_x = res.x
    else:
        # Fallback to a known good point or random
        # The seed program's fallback was [-1.13, 0.04, 0.0003]
        # Let's try a few random points if DE fails
        best_val = 1e15
        best_x = np.array([-1.13, 0.04, 0.0003])
        for _ in range(10):
            x_try = np.random.uniform(-5, 5, 3)
            val_try = objective(x_try)
            if val_try < best_val:
                best_val = val_try
                best_x = x_try
        # Polish the best found
        try:
            res_polish = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 1000})
            if res_polish.fun < best_val:
                best_x = res_polish.x
        except:
            pass

    return best_x

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
