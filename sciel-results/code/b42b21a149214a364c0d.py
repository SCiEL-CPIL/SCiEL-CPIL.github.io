"""
Symbolic regression: recover an unknown function f(x) with 1 input from [-3, 3].
We design a flexible basis that BFGS can fit well with 10 parameters.
Strategy: use a rich set of basis functions that can represent many common
smooth functions: polynomials, trigonometric, rational, exponential, sigmoid, etc.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model with 10 parameters using a diverse basis set.
    
    Basis functions chosen to cover:
    - Constant, linear, quadratic, cubic, quartic (polynomial up to degree 4)
    - Sine, cosine (periodic behavior)
    - Tanh (sigmoid-like)
    - 1/(1+x^2) (Lorentzian / rational)
    - exp(-x^2) (Gaussian bump)
    
    This gives 10 linearly independent (mostly) basis functions, and the 10
    parameters are linear coefficients. BFGS will fit these efficiently since
    the model is linear in parameters (convex optimization in p).
    
    Actually, since the model is linear in params, BFGS should converge very well.
    Let's use a set of basis functions that spans a large space of smooth functions.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is at least 1-D
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Build basis matrix: (N, 10)
    # Basis 0: 1 (constant)
    # Basis 1: x
    # Basis 2: x^2
    # Basis 3: x^3
    # Basis 4: x^4
    # Basis 5: sin(x)
    # Basis 6: cos(x)
    # Basis 7: tanh(x)
    # Basis 8: 1/(1 + x^2)
    # Basis 9: exp(-x^2)
    
    b0 = np.ones_like(x_flat)
    b1 = x_flat
    b2 = x_flat ** 2
    b3 = x_flat ** 3
    b4 = x_flat ** 4
    b5 = np.sin(x_flat)
    b6 = np.cos(x_flat)
    b7 = np.tanh(x_flat)
    b8 = 1.0 / (1.0 + x_flat ** 2)
    b9 = np.exp(-x_flat ** 2)
    
    # Combine
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
              p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    return result.reshape(orig_shape)
# EVOLVE-BLOCK-END
