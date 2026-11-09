"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-Informed Polynomial Model for Nonlinear Oscillator.
    
    Uses a Duffing-van der Pol inspired basis with higher-order terms.
    The model is linear in the parameters, which makes the optimization
    landscape convex and allows BFGS to find the global optimum reliably.
    
    Basis functions:
    1. 1
    2. x
    3. x^3
    4. v
    5. v^3
    6. x^2 v
    7. x v
    8. x^2
    9. v^2
    10. x^4
    
    Inputs:
      x: position array
      v: velocity array
      params: array of 10 coefficients
      
    Returns:
      a: acceleration array
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure p has length 10, pad with zeros if necessary (though evaluator passes 10)
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
        
    # Compute basis terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    
    # Linear combination
    # p[0]: bias
    # p[1]: x
    # p[2]: x^3
    # p[3]: v
    # p[4]: v^3
    # p[5]: x^2 v
    # p[6]: x v
    # p[7]: x^2
    # p[8]: v^2
    # p[9]: x^4
    
    a = (p[0] * 1.0 + 
         p[1] * x + 
         p[2] * x3 + 
         p[3] * v + 
         p[4] * v3 + 
         p[5] * x2 * v + 
         p[6] * x * v + 
         p[7] * x2 + 
         p[8] * v2 + 
         p[9] * x4)
         
    return a
# EVOLVE-BLOCK-END
