import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Linear Basis: polynomial + exponential + trigonometric terms.
    
    The model is linear in parameters, so BFGS (or any linear solver) can find
    the optimal coefficients very reliably. The basis is chosen to cover:
    - Constant offset
    - Linear and quadratic trends
    - Exponential growth/decay
    - Periodic oscillations at frequencies 1, 2, 3
    
    Parameters (10 total):
    p[0]: constant
    p[1]: coefficient for x
    p[2]: coefficient for x^2
    p[3]: coefficient for exp(x)
    p[4]: coefficient for sin(x)
    p[5]: coefficient for cos(x)
    p[6]: coefficient for sin(2x)
    p[7]: coefficient for cos(2x)
    p[8]: coefficient for sin(3x)
    p[9]: coefficient for cos(3x)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        # Fallback: return ones * p[0] if shape is wrong
        return np.ones_like(x) * (p[0] if p.size > 0 else 0.0)
    
    # Compute basis functions
    # Scale x for polynomial terms to keep them in a reasonable range [0.1, 6]
    # No scaling needed for exp/sin/cos since they are bounded or well-behaved in this range
    
    term_1 = np.ones_like(x)
    term_2 = x
    term_3 = x ** 2
    term_4 = np.exp(x)
    term_5 = np.sin(x)
    term_6 = np.cos(x)
    term_7 = np.sin(2.0 * x)
    term_8 = np.cos(2.0 * x)
    term_9 = np.sin(3.0 * x)
    term_10 = np.cos(3.0 * x)
    
    # Linear combination
    result = (p[0] * term_1 + 
              p[1] * term_2 + 
              p[2] * term_3 + 
              p[3] * term_4 + 
              p[4] * term_5 + 
              p[5] * term_6 + 
              p[6] * term_7 + 
              p[7] * term_8 + 
              p[8] * term_9 + 
              p[9] * term_10)
    
    return result
