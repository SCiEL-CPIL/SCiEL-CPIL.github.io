"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a sum of basis functions:
    - Constant
    - Linear
    - Quadratic
    - Cubic
    - Exponential (exp(a*x) - 1)
    - Logarithmic (log(x))
    - Sine (sin(x))
    - Cosine (cos(x))
    - Reciprocal (1/x)
    - x^4
    
    params: array of length 10, each corresponding to a coefficient.
    """
    # Ensure x is 1D array
    x = np.asarray(x, dtype=np.float64)
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = np.exp(params[8] * x) - 1.0  # exponential with a tunable rate inside? No, params are coefficients.
    # Wait, the contract says "fits params, an array of length 10". The params are just coefficients.
    # So I should define fixed basis functions and multiply by params.
    
    # Let's redefine basis functions to be fixed, and params are just linear coefficients.
    # This makes it a linear regression problem essentially, which BFGS can solve well if the basis is good.
    # However, BFGS is a general optimizer. If the function is linear in params, BFGS will converge quickly.
    
    # Basis set:
    # 1. 1
    # 2. x
    # 3. x^2
    # 4. x^3
    # 5. x^4
    # 6. exp(x)
    # 7. log(x)
    # 8. sin(x)
    # 9. cos(x)
    # 10. 1/x
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = np.exp(x)
    b6 = np.log(x)
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = 1.0 / x
    
    # Combine
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
