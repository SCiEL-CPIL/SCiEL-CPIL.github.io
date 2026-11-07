import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Chebyshev polynomial (degree 4) + Gaussian bumps.
    
    5 parameters: Chebyshev coefficients [c0, c1, c2, c3, c4]
    5 parameters: Gaussian centers [m1, m2, m3, m4, m5]
    
    The Gaussian amplitudes and widths are fixed (1.0 and 0.25) to avoid
    the optimization difficulties of having too many free parameters.
    This provides a global polynomial approximation plus localized bumps.
    """
    # Deadline guard
    t0 = time.time()
    
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Split parameters
    cheb_coeffs = params[0:5]
    centers = params[5:10]
    
    # Evaluate Chebyshev polynomial of degree 4
    # T0=1, T1=t, T2=2t^2-1, T3=4t^3-3t, T4=8t^4-8t^2+1
    y = np.zeros_like(x)
    y += cheb_coeffs[0] * np.ones_like(x)
    y += cheb_coeffs[1] * t
    y += cheb_coeffs[2] * (2.0 * t**2 - 1.0)
    y += cheb_coeffs[3] * (4.0 * t**3 - 3.0 * t)
    y += cheb_coeffs[4] * (8.0 * t**4 - 8.0 * t**2 + 1.0)
    
    # Add Gaussian bumps
    for i in range(5):
        y += np.exp(-((x - centers[i]) ** 2) / (2.0 * 0.25**2))
    
    # Safety: check for non-finite values
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
    
    return y
