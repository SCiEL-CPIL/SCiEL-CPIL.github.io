# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    Uses a full cubic polynomial in standardized coordinates X and V.
    Standardization maps the known training ranges to [-1, 1].
    This improves the conditioning of the design matrix for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Standardize inputs to [-1, 1] based on training ranges
    # x range: [-0.5728, 0.5275]
    # v range: [-0.1754, 0.2821]
    
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    X = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    V = 2.0 * (v - v_min) / (v_max - v_min) - 1.0
    
    # Full cubic polynomial in X, V
    # p[0]: 1
    # p[1]: X
    # p[2]: V
    # p[3]: X^2
    # p[4]: V^2
    # p[5]: X*V
    # p[6]: X^3
    # p[7]: V^3
    # p[8]: X^2*V
    # p[9]: X*V^2
    
    a = (p[0] +
         p[1] * X +
         p[2] * V +
         p[3] * X**2 +
         p[4] * V**2 +
         p[5] * X * V +
         p[6] * X**3 +
         p[7] * V**3 +
         p[8] * X**2 * V +
         p[9] * X * V**2)
    
    return a
# EVOLVE-BLOCK-END
