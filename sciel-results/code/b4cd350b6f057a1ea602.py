"""Symbolic regression with a rich, stable linear basis for BFGS-compatible optimization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a linear combination of 10 basis functions.
    
    The basis functions are chosen to be diverse and well-behaved on [-2, 2]:
    - Polynomials: 1, x, x^2, x^3, x^4
    - Trig: sin(x), cos(x), sin(2x), cos(2x)
    - Sigmoid: tanh(x)
    
    This linear parameterization ensures the loss surface is convex with respect to 
    the parameters (for fixed x), allowing BFGS to converge reliably to the global 
    minimum for this specific function space.
    """
    # Deadline guard: The evaluator kills after 150s. 
    # This function is called repeatedly by the optimizer. 
    # We don't track time inside equation() because it's stateless and fast.
    # The deadline guard is effectively handled by the speed of the computation.
    # However, we ensure no infinite loops or heavy computations here.
    
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure exactly 10 parameters; pad or truncate if needed
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
        
    # Compute basis functions
    # 1. Polynomials
    b0 = 1.0
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    
    # 2. Trigonometric
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    
    # 3. Hyperbolic tangent
    b9 = np.tanh(x)
    
    # Linear combination
    result = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
              
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
