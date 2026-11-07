# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: a rich polynomial basis in x and y up to degree 3,
    with 10 learnable coefficients. The basis is fixed (closed-form), and BFGS fits the
    10 params from the training data.
    
    Basis functions (10 total):
      1. 1
      2. x
      3. y
      4. x^2
      5. y^2
      6. x*y
      7. x^3
      8. y^3
      9. x^2*y
      10. x*y^2
    
    This allows recovery of any polynomial up to degree 3 in two variables, which is a
    much richer class than the current linear model. For many common hidden functions
    (polynomials, or functions well-approximated by low-degree polynomials on [-1,1]^2),
    this should significantly reduce MSE.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis values
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Linear combination of basis functions
    pred = (p0 
            + p1 * x 
            + p2 * y 
            + p3 * x2 
            + p4 * y2 
            + p5 * xy 
            + p6 * x3 
            + p7 * y3 
            + p8 * x2y 
            + p9 * xy2)
    
    return pred
# EVOLVE-BLOCK-END
