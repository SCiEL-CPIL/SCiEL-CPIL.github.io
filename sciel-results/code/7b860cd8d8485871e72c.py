"""
Symbolic regression: recover an unknown function f(x) with 1 input from [-3, 3].
Strategy: Use a rich, diverse basis set of 10 functions that spans a large space
of smooth functions. The model is linear in parameters, so BFGS converges to the
global least-squares solution efficiently.

Basis functions:
- Chebyshev polynomials (T0-T4) for well-conditioned polynomial approximation
- Trigonometric terms (sin(x), cos(x), sin(2x), cos(2x)) for periodic behavior
- Sigmoid (tanh(x)) for step-like transitions
- Gaussian (exp(-x^2)) for localized bumps
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model with 10 parameters using a well-conditioned basis set.
    
    Basis functions:
    0: T0(x) = 1 (constant)
    1: T1(x) = x_scaled (linear in scaled x)
    2: T2(x) = 2*x_scaled^2 - 1 (quadratic Chebyshev)
    3: T3(x) = 4*x_scaled^3 - 3*x_scaled (cubic Chebyshev)
    4: T4(x) = 8*x_scaled^4 - 8*x_scaled^2 + 1 (quartic Chebyshev)
    5: sin(x)
    6: cos(x)
    7: sin(2x)
    8: cos(2x)
    9: exp(-x^2)
    
    Chebyshev polynomials are scaled to [-1, 1] for better conditioning.
    The model is linear in params, ensuring convex loss and reliable BFGS convergence.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is at least 1-D
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Scale x to [-1, 1] for Chebyshev polynomials
    x_scaled = x_flat / 3.0
    
    # Build basis matrix: (N, 10)
    # Chebyshev polynomials T_n(x_scaled)
    b0 = np.ones_like(x_flat)
    b1 = x_scaled
    b2 = 2.0 * x_scaled ** 2 - 1.0
    b3 = 4.0 * x_scaled ** 3 - 3.0 * x_scaled
    b4 = 8.0 * x_scaled ** 4 - 8.0 * x_scaled ** 2 + 1.0
    # Trigonometric terms
    b5 = np.sin(x_flat)
    b6 = np.cos(x_flat)
    b7 = np.sin(2.0 * x_flat)
    b8 = np.cos(2.0 * x_flat)
    # Gaussian
    b9 = np.exp(-x_flat ** 2)
    
    # Combine
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
              p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    return result.reshape(orig_shape)
# EVOLVE-BLOCK-END
