"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A rich basis expansion that is linear in parameters.
    The evaluator fits params (length 10) using BFGS starting from ones.
    We design the basis so that the optimal params are close to 1 (or at least
    BFGS can find a good fit quickly).
    
    Since the target is unknown, we use a general-purpose basis that spans
    polynomials up to degree 5, plus some trigonometric terms, to cover
    common symbolic regression targets.
    
    Basis functions (10 total):
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: x^5
    6: sin(x)
    7: cos(x)
    8: exp(x)  -- might be too large, let's use exp(-x^2) or something bounded
    9: |x| or tanh(x)
    
    Actually, exp(x) can overflow or be huge. Let's use:
    8: exp(-x^2)
    9: tanh(x)
    
    But wait, the params are fit by BFGS. If the basis functions have very different
    scales, BFGS might struggle. However, since we start from ones and BFGS is
    robust for smooth problems, this should be okay.
    
    Let's ensure numerical stability.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Precompute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Basis functions, normalized to have max abs value ~1 over [-3, 3]
    b0 = np.ones_like(x) / 1.0
    b1 = x / 3.0
    b2 = x2 / 9.0
    b3 = x3 / 27.0
    b4 = x4 / 81.0
    b5 = x5 / 243.0
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.exp(-x2)
    b9 = np.tanh(x)
    
    # Linear combination
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
