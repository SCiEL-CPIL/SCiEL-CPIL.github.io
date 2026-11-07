# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis:
    - 5 polynomial terms: 1, x, x^2, x^3, x^4
    - 4 trigonometric terms with learnable frequencies:
      sin(w1*x), cos(w1*x), sin(w2*x), cos(w2*x)
    - 1 Gaussian RBF term with learnable center and width:
      exp(-((x - c)^2) / (2 * sigma^2))
    
    params layout (length 10):
      p[0]: coeff for 1 (bias)
      p[1]: coeff for x
      p[2]: coeff for x^2
      p[3]: coeff for x^3
      p[4]: coeff for x^4
      p[5]: coeff for sin(w1*x)
      p[6]: coeff for cos(w1*x)
      p[7]: w1 (frequency)
      p[8]: coeff for sin(w2*x)
      p[9]: coeff for cos(w2*x)
    
    Wait, that's 10 params but no RBF. Let me restructure to include RBF.
    
    Actually, the current program uses 9 RBFs + 1 width = 10 params.
    The inspiration uses poly + trig.
    
    Let's combine: Use a fixed set of basis functions that are linear in the
    coefficients, but include a few learnable parameters for flexibility.
    
    Better approach: Use a fixed rich basis that is linear in all 10 parameters.
    This makes the problem convex and BFGS will converge to the global optimum
    for the given basis. The key is choosing a basis that spans a wide class
    of functions.
    
    Basis: 1, x, x^2, x^3, x^4, sin(x), cos(x), sin(2x), cos(2x), exp(-x^2/2)
    
    All 10 parameters are coefficients. The basis is fixed.
    This is linear in params -> convex optimization -> BFGS finds global min.
    
    Let's try this.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Build fixed basis matrix: N x 10
    # b0 = 1
    # b1 = x
    # b2 = x^2
    # b3 = x^3
    # b4 = x^4
    # b5 = sin(x)
    # b6 = cos(x)
    # b7 = sin(2x)
    # b8 = cos(2x)
    # b9 = exp(-x^2/2)
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    b9 = np.exp(-x2 / 2.0)
    
    # Stack into matrix
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    result = B @ params
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
