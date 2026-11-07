"""Rewritten program for symbolic regression with a rational/ratio basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a basis of linear, inverse, and ratio terms.
    Basis: [1, x, y, z, 1/x, 1/y, 1/z, x/y, y/z, z/x]
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Prevent division by zero (though domain is [0.5, 2], this is a safety net)
    # Using np.where to avoid warnings
    safe_x = np.where(np.abs(x) < 1e-12, 1e-12, x)
    safe_y = np.where(np.abs(y) < 1e-12, 1e-12, y)
    safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    
    # Compute basis terms
    term_0 = np.ones_like(x)
    term_1 = x
    term_2 = y
    term_3 = z
    term_4 = 1.0 / safe_x
    term_5 = 1.0 / safe_y
    term_6 = 1.0 / safe_z
    term_7 = x / safe_y
    term_8 = y / safe_z
    term_9 = z / safe_x
    
    # Combine with parameters
    # params[0] * term_0 + ... + params[9] * term_9
    prediction = (
        params[0] * term_0 +
        params[1] * term_1 +
        params[2] * term_2 +
        params[3] * term_3 +
        params[4] * term_4 +
        params[5] * term_5 +
        params[6] * term_6 +
        params[7] * term_7 +
        params[8] * term_8 +
        params[9] * term_9
    )
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(prediction)):
        # Return large values to penalize during fitting if it happens
        return np.full_like(prediction, 1e30)
        
    return prediction
# EVOLVE-BLOCK-END
