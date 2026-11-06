"""Improved program for the suite task; fits an odd-degree polynomial closure exploiting antisymmetry."""
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
    if len(x) < 3:
        return np.zeros(6)
        
    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)
        
    try:
        # The unresolved fast-scale forcing in the two-scale Lorenz-96 model is an odd function
        # of the local slow state x (due to the antisymmetry of the fast-slow coupling).
        # Therefore, we restrict the polynomial to odd powers only: x, x^3, x^5.
        # This reduces the effective degrees of freedom from 6 to 3, improving generalization
        # to the unseen F=10 regime.
        
        # Construct the design matrix for odd powers: [x, x^3, x^5]
        # We use np.vander with increasing=False and then select odd columns, 
        # or simply construct it directly for clarity and speed.
        # x^1, x^3, x^5
        A = np.column_stack((x, x**3, x**5))
        
        # Solve the least squares problem A @ coeffs_odd = y
        # np.linalg.lstsq returns the minimum-norm solution in case of rank deficiency
        coeffs_odd, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        
        # Ensure coefficients are finite
        if not np.all(np.isfinite(coeffs_odd)):
            # Fallback to linear fit (odd power x) if high-degree fit is unstable
            # Using polyfit for consistency with previous fallbacks, or just lstsq on x
            A_lin = x.reshape(-1, 1)
            coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            coeffs = np.zeros(6)
            coeffs[1] = coeffs_lin[0]
            return coeffs
            
        # Map the 3 odd coefficients to the 6-element output array (ascending powers)
        # coeffs[0] = c0 (x^0) = 0
        # coeffs[1] = c1 (x^1)
        # coeffs[2] = c2 (x^2) = 0
        # coeffs[3] = c3 (x^3)
        # coeffs[4] = c4 (x^4) = 0
        # coeffs[5] = c5 (x^5)
        coeffs = np.zeros(6)
        coeffs[1] = coeffs_odd[0]
        coeffs[3] = coeffs_odd[1]
        coeffs[5] = coeffs_odd[2]
        
        return coeffs
        
    except Exception:
        # Fallback to linear fit in case of any errors
        try:
            A_lin = x.reshape(-1, 1)
            coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
            coeffs = np.zeros(6)
            coeffs[1] = coeffs_lin[0]
            return coeffs
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
