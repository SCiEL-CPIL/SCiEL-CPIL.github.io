"""Rewritten program using global optimization for the uncertainty inequality task."""
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

    # Define bounds for the search space.
    # The coefficients can be scaled arbitrarily, so we fix a scale.
    # However, the objective is scale-invariant.
    # We can restrict the search to a sphere or a box.
    # Let's use a box [-10, 10] for each coefficient.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use differential evolution for global search
    # Then refine with local optimization
    result = differential_evolution(objective, bounds, seed=42, maxiter=1000, tol=1e-12, polish=True)
    
    if result.success:
        best_x = result.x
    else:
        # Fallback if DE fails
        best_x = np.array([-1.0, 0.1, 0.01])
        
    # Refine with L-BFGS-B starting from the DE result
    try:
        res = minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15})
        if res.success and res.fun < objective(best_x):
            best_x = res.x
    except Exception:
        pass
        
    return best_x

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
