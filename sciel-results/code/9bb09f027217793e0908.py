"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a degree-5 polynomial to the unresolved tendency using ridge regression.
    We use a small regularization to prevent overfitting while allowing the full
    degree-5 basis to capture non-linearities better than the linear seed.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge cases
    if len(x) == 0:
        return np.zeros(6)
    
    # Build Vandermonde matrix for polynomial fit (ascending powers)
    # V[:, i] = x^i
    V = np.vander(x, N=6, increasing=True)
    
    # Ridge regression: minimize ||Vc - y||^2 + lambda * ||c||^2
    # This is equivalent to solving (V^T V + lambda I) c = V^T y
    
    # Choose a small lambda. 
    # Since the task asks for a shared polynomial, and the data might be noisy,
    # a small regularization helps stability.
    # We can tune this, but a small fixed value is a good start.
    # Let's try lambda = 1e-4 or similar. 
    # Actually, let's check the scale of y.
    # If y is large, lambda should be scaled? 
    # Standard ridge uses lambda * I.
    # Let's try a few values or just pick a reasonable one.
    # The seed was linear (effectively high regularization on higher terms).
    # Full degree 5 with no regularization might overfit.
    # Let's use lambda = 1e-3 * (variance of y) to make it scale invariant?
    # Or just a small constant.
    
    # Let's try lambda = 1e-4
    lam = 1e-4
    
    # Add regularization to diagonal
    # V^T V is 6x6
    A = V.T @ V
    b = V.T @ y
    
    # Add lambda to diagonal
    A_reg = A + lam * np.eye(6)
    
    # Solve
    try:
        coeff = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular (unlikely with ridge, but safe)
        coeff = np.linalg.lstsq(V, y, rcond=None)[0]
        # Ensure length 6
        if len(coeff) < 6:
            coeff = np.pad(coeff, (0, 6-len(coeff)))
        else:
            coeff = coeff[:6]
            
    # Ensure finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to linear fit
        coeff = np.zeros(6)
        if len(x) > 1:
            c1 = np.polyfit(x, y, 1) # descending order
            coeff[0] = c1[1]
            coeff[1] = c1[0]
        else:
            coeff[0] = np.mean(y)
            
    return coeff
# EVOLVE-BLOCK-END
