"""Rewritten program using global optimization for the uncertainty inequality task."""
import numpy as np
from scipy.optimize import differential_evolution

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
        
        return (X * K) / (N * N)

    # Define bounds for the coefficients
    # The problem is scale invariant, so we can restrict the scale, 
    # but differential_evolution works on the defined bounds.
    # We can set reasonable bounds. Since the function is homogeneous of degree 0,
    # the magnitude doesn't matter, only the ratios.
    # We can fix one variable or just search in a box.
    # Let's search in [-1, 1] for all, which covers all directions on the sphere.
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    
    # Run differential evolution
    # maxiter=1000, popsize=15 should be sufficient for 3 variables
    res = differential_evolution(objective, bounds, maxiter=1000, popsize=20, tol=1e-12, seed=42)
    
    if res.success:
        return res.x
    else:
        # Fallback to a known good point if optimization fails
        # The seed program returned something, let's try to recover or use a default
        return np.array([-113/100, 1/25, 1/3240])

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
