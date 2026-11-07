"""Symbolic regression with 1 input using a rich basis + internal least-squares refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a rich basis of 10 well-conditioned functions spanning [0.1, 6].
    The model is linear in params, so we can do an internal least-squares
    refinement to get a better fit than BFGS from all-ones might achieve.
    
    Since we don't have access to y, we can't do true least squares internally.
    But we CAN use the params as a starting point and do a few Newton steps
    if we had gradients. However, without y, we can't compute residuals.
    
    So the best we can do is return the linear combination with a well-chosen
    basis that BFGS can fit well from all-ones.
    
    Key insight: use a basis that is orthogonal or near-orthogonal on [0.1, 6]
    so that BFGS converges quickly. Chebyshev polynomials mapped to the domain
    are a good choice.
    
    But we can also try a hybrid: use some polynomial terms and some
    exponential/sine terms to capture non-polynomial behavior.
    
    Let's use:
    - 1 (constant)
    - x, x^2, x^3, x^4 (polynomial up to degree 4)
    - sin(x), cos(x), sin(2x), cos(2x) (periodic components)
    - exp(-x) (decaying exponential)
    
    This gives 10 basis functions that span a diverse function space.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build basis matrix
    n = len(x)
    basis = np.zeros((n, 10), dtype=np.float64)
    
    # Polynomial terms
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    basis[:, 4] = x ** 4
    
    # Periodic terms
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.cos(x)
    basis[:, 7] = np.sin(2.0 * x)
    basis[:, 8] = np.cos(2.0 * x)
    
    # Exponential decay
    basis[:, 9] = np.exp(-x)
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
