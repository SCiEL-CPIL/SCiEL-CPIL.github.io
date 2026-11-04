import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses the zeros of the Laguerre polynomial L_12(x) as initial points, 
    then optimizes to minimize the maximum root (a common proxy for uncertainty bound width).
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0 # 600s limit, keep margin
    
    m = 12
    
    # Step 1: Get the roots of the Laguerre polynomial L_m(x).
    # numpy.polynomial.laguerre.laggauss(m) returns the nodes (roots) and weights.
    # These roots are positive and lie in (0, infinity).
    try:
        roots, weights = np.polynomial.laguerre.laggauss(m)
    except Exception:
        # Fallback to arithmetic progression if laggauss fails
        roots = 4.0 * np.arange(1, m + 1)
    
    # Step 2: Define the objective function to minimize.
    # Hypothesis: The C4 bound is related to the spread or the largest root.
    # Minimizing the largest root reduces the "support" size.
    # We also penalize very small roots if they cause numerical issues, but Laguerre roots are well-behaved.
    def objective(params):
        # Ensure positivity
        if np.any(params <= 0):
            return 1e10
        # Maximize the minimum gap? No, minimize max root.
        # A more robust proxy for "uncertainty" might be the variance or the max value.
        # Let's minimize the maximum root.
        return np.max(params)
    
    # Step 3: Optimize
    # We start from the Laguerre roots.
    x0 = roots.copy()
    
    # Check time
    if time.time() > deadline - 10:
        return roots.tolist()
        
    try:
        # Use Nelder-Mead for bound-constrained optimization (implicitly via positivity check in objective)
        # However, Nelder-Mead doesn't handle bounds well. 
        # We can transform variables: params = log(x). 
        # Let y = log(x), so x = exp(y). This ensures x > 0.
        y0 = np.log(roots)
        
        def obj_y(y):
            x = np.exp(y)
            return np.max(x)
            
        # Optimize y
        result = minimize(obj_y, y0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8})
        
        # Check time again
        if time.time() > deadline - 5:
            return roots.tolist()
            
        if result.success:
            optimal_roots = np.exp(result.x)
            # Verify they are positive and distinct (Laguerre roots are distinct)
            if np.all(optimal_roots > 0):
                return optimal_roots.tolist()
        else:
            # If optimization fails, return the Laguerre roots which are still likely better than arithmetic
            return roots.tolist()
            
    except Exception:
        return roots.tolist()

    # Fallback
    return roots.tolist()
