"""Rewritten program for symbolic regression with quadratic basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a quadratic basis expansion.
    
    The basis consists of 10 terms:
    1, x_c, y_c, z_c, x_c^2, y_c^2, z_c^2, x_c*y_c, x_c*z_c, y_c*z_c
    where x_c, y_c, z_c are centered inputs.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
        
    # Center inputs to improve numerical conditioning for BFGS
    # Mean of [0.5, 2] is 1.25
    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25
    
    # Construct the feature matrix (n_samples, 10)
    # Modified basis: Replace yc*zc with xc*yc*zc to capture 3-way interactions
    # This is a hypothesis that the target function has a significant xyz component
    features = np.column_stack((
        np.ones_like(xc),
        xc,
        yc,
        zc,
        xc * xc,
        yc * yc,
        zc * zc,
        xc * yc,
        xc * zc,
        xc * yc * zc
    ))
    
    # Compute prediction: features (n, 10) dot params (10,) -> (n,)
    pred = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # If numerical issues arise, fall back to a safer linear model or clip
        # However, with centered inputs, this is unlikely.
        # For safety, replace non-finite with 0
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
