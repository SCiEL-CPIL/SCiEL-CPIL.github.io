"""
Strategy: Linear Basis with Trigonometric and Polynomial Terms.

The previous quadratic polynomial basis scored worse than the seed, suggesting
the target function may not be well-approximated by a quadratic surface, or
the optimization/conditioning of the quadratic basis was suboptimal.
A basis combining linear terms and trigonometric functions (sin/cos) provides
a smoother, more flexible approximation for functions on the bounded domain
[0.5, 2]. This basis is still linear in the parameters, ensuring the
optimization landscape is convex and BFGS will find the global minimum for
this specific basis. The trigonometric terms allow the model to capture
oscillatory or non-polynomial curvature that a pure polynomial might miss.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    f(x,y,z) = c0 + c1*x + c2*y + c3*z + 
               c4*sin(x) + c5*sin(y) + c6*sin(z) + 
               c7*cos(x) + c8*cos(y) + c9*cos(z)
    
    Parameters (10 total):
    params[0]: c0 (constant)
    params[1]: c1 (x coefficient)
    params[2]: c2 (y coefficient)
    params[3]: c3 (z coefficient)
    params[4]: c4 (sin(x) coefficient)
    params[5]: c5 (sin(y) coefficient)
    params[6]: c6 (sin(z) coefficient)
    params[7]: c7 (cos(x) coefficient)
    params[8]: c8 (cos(y) coefficient)
    params[9]: c9 (cos(z) coefficient)
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Compute basis functions
    term0 = c0
    term1 = c1 * x
    term2 = c2 * y
    term3 = c3 * z
    
    term4 = c4 * np.sin(x)
    term5 = c5 * np.sin(y)
    term6 = c6 * np.sin(z)
    
    term7 = c7 * np.cos(x)
    term8 = c8 * np.cos(y)
    term9 = c9 * np.cos(z)
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
