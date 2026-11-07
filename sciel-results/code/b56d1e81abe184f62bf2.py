"""
Symbolic regression: hybrid polynomial and tanh basis with 10 features.
Combines low-order polynomial terms with bounded tanh nonlinearities and
cross-interaction terms to capture a wide range of functional forms on [-1,1]x[-1,1].
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions for given x, y, params.
    
    params has length 10, corresponding to:
        1, x, y, x^2, y^2, xy, tanh(x), tanh(y), x*tanh(y), y*tanh(x)
    
    This basis includes low-order polynomials and tanh nonlinearities
    with cross-interaction terms to capture polynomial, saturating,
    and coupled functional forms.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute tanh values (bounded, smooth, captures saturation)
    tx = np.tanh(x)
    ty = np.tanh(y)
    
    # Compute predictions
    pred = (
        p0 * np.ones_like(x) +
        p1 * x +
        p2 * y +
        p3 * x**2 +
        p4 * y**2 +
        p5 * x * y +
        p6 * tx +
        p7 * ty +
        p8 * x * ty +
        p9 * y * tx
    )
    
    # Ensure the output is a finite array with correct shape
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something went wrong
        pred = p0 * np.ones_like(x) + p1 * x + p2 * y
    
    return pred
# EVOLVE-BLOCK-END
