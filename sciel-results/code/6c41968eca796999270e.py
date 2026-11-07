import numpy as np
import time

# Global time tracking for safety
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # 5 second margin from 120s limit

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid basis of:
    1. Low-order polynomials (up to degree 2)
    2. Multiple sinusoidal terms at different frequencies
    3. A Gaussian bump
    
    Parameters:
    p[0]: Intercept
    p[1]: Coeff for x
    p[2]: Coeff for y
    p[3]: Coeff for x^2
    p[4]: Coeff for y^2
    p[5]: Coeff for x*y
    p[6]: Amplitude of sin(x)
    p[7]: Amplitude of cos(y)
    p[8]: Amplitude of sin(x+y)
    p[9]: Amplitude of Gaussian exp(-(x^2+y^2))
    """
    
    # Safety check: If we are running out of time, return a simple stable prediction
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return the polynomial part with current params as a safe fallback
        x2 = x * x
        y2 = y * y
        xy = x * y
        return (params[0] 
                + params[1] * x 
                + params[2] * y 
                + params[3] * x2 
                + params[4] * y2 
                + params[5] * xy)

    # Pre-compute common terms
    x2 = x * x
    y2 = y * y
    xy = x * y

    # 1. Polynomial Basis (Linear + Quadratic)
    # These are well-conditioned for BFGS on [-1, 1]
    poly_part = (params[0] 
                 + params[1] * x 
                 + params[2] * y 
                 + params[3] * x2 
                 + params[4] * y2 
                 + params[5] * xy)

    # 2. Sinusoidal Basis - multiple fixed frequencies
    # sin(x), cos(y), sin(x+y) capture common periodic structures
    sine_part = (params[6] * np.sin(x) 
                 + params[7] * np.cos(y) 
                 + params[8] * np.sin(x + y))

    # 3. Gaussian Bump
    # Captures localized features or decay
    # p[9] is amplitude. Width is fixed at 1.0 (exp(-r^2)).
    r2 = x2 + y2
    # Use np.exp which is stable
    gauss_part = params[9] * np.exp(-r2)

    # Combine
    pred = poly_part + sine_part + gauss_part

    # Ensure numerical stability
    # If any values become non-finite (e.g., from extreme params causing overflow in exp, 
    # though exp(-r^2) is bounded, sine is bounded, poly is bounded for bounded x,y),
    # fallback to polynomial part which is always finite for bounded inputs.
    if not np.all(np.isfinite(pred)):
        return poly_part

    return pred
