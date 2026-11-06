"""Improved program for the suite task; fits a robust degree-five polynomial closure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge case of empty or tiny training sets
    if len(x) < 6:
        return np.zeros(6)
        
    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)
        
    try:
        # Use np.polynomial.polynomial.polyfit for least-squares fit of degree 5
        # This uses SVD and is numerically stable for moderate degrees
        coeffs = np.polynomial.polynomial.polyfit(x, y, 5)
        
        # Ensure coefficients are finite
        if not np.all(np.isfinite(coeffs)):
            # Fallback to linear fit if high-degree fit is unstable
            coeffs_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeffs = np.zeros(6)
            coeffs[:2] = coeffs_lin
            return coeffs
            
        return coeffs
        
    except Exception:
        # Fallback to linear fit in case of any errors
        try:
            coeffs_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeffs = np.zeros(6)
            coeffs[:2] = coeffs_lin
            return coeffs
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
