"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a rich basis of common functional forms.
    
    We construct a feature matrix with many candidate basis functions of (x, y),
    then return a linear combination with the 10 learnable params.
    Since BFGS optimizes params starting from all ones, we design the basis
    so that the identity mapping (or common simple functions) is approximately
    representable with params near 1, but also has enough flexibility.
    
    Actually, since BFGS starts from all ones and optimizes, we want the basis
    to be such that a good fit is achievable. Let's use a combination of:
    - Polynomial terms up to degree 3 in x and y
    - Trigonometric terms
    - Rational terms
    - Exponential terms (bounded)
    - tanh terms
    
    But we only have 10 params. So we need to pick 10 basis functions.
    
    Let's think about what common hidden functions might be:
    - x^2, y^2, xy, x+y, x*y, sin(x), cos(y), exp(x), etc.
    
    Strategy: Use 10 diverse basis functions that span common symbolic forms.
    
    Basis functions (each returns an array of same shape as x):
    0: x
    1: y
    2: x^2
    3: y^2
    4: x*y
    5: x^3
    6: y^3
    7: sin(x)
    8: sin(y)
    9: tanh(x*y)  # or some interaction
    
    But wait - BFGS starts params at all ones. If the true function is, say, f(x,y) = x^2 + y^2,
    then with our basis, the optimal params would be [0,0,1,1,0,0,0,0,0,0]. Starting from all ones,
    BFGS needs to drive 8 params to 0 and keep 2 at 1. This should work if the basis is well-conditioned.
    
    However, a better approach: include a constant term (bias) since many functions have offsets.
    Let's replace one basis with 1.0.
    
    Revised basis:
    0: 1.0 (bias)
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: x^3
    7: y^3
    8: sin(x)
    9: sin(y)
    
    This covers polynomials up to degree 3 and basic trig. Should handle many common symbolic regression targets.
    """
    # Ensure x, y are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Build basis functions
    basis = np.column_stack([
        np.ones_like(x),      # 0: bias
        x,                     # 1
        y,                     # 2
        x**2,                  # 3
        y**2,                  # 4
        x * y,                 # 5
        x**3,                  # 6
        y**3,                  # 7
        np.sin(x),             # 8
        np.sin(y),             # 9
    ])
    
    # params is a 1D array of length 10
    params = np.asarray(params, dtype=np.float64).flatten()
    if params.shape[0] != 10:
        # Fallback: pad or truncate
        if params.shape[0] < 10:
            params = np.pad(params, (0, 10 - params.shape[0]), constant_values=0)
        else:
            params = params[:10]
    
    # Linear combination
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
