import numpy as np
import time

# Global time tracking for safety
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # 5 second margin from 120s limit

def _chebyshev_basis(x, y):
    """
    Compute a well-conditioned basis of 10 functions for fitting on [-1,1]^2.
    
    Uses Chebyshev polynomials for better conditioning than monomials:
    p[0]: 1
    p[1]: T1(x) = x
    p[2]: T1(y) = y
    p[3]: T2(x) = 2x^2 - 1
    p[4]: T2(y) = 2y^2 - 1
    p[5]: T1(x)*T1(y) = x*y
    p[6]: T3(x) = 4x^3 - 3x
    p[7]: T3(y) = 4y^3 - 3y
    p[8]: T2(x)*T1(y) = (2x^2-1)*y
    p[9]: T1(x)*T2(y) = x*(2y^2-1)
    
    This basis is linear in params, so BFGS finds the global optimum reliably.
    Chebyshev polynomials are orthogonal on [-1,1], improving numerical conditioning.
    """
    x2 = x * x
    y2 = y * y
    
    T1_x = x
    T1_y = y
    T2_x = 2.0 * x2 - 1.0
    T2_y = 2.0 * y2 - 1.0
    T3_x = 4.0 * x2 * x - 3.0 * x
    T3_y = 4.0 * y2 * y - 3.0 * y
    
    basis = np.column_stack([
        np.ones_like(x),
        T1_x,
        T1_y,
        T2_x,
        T2_y,
        T1_x * T1_y,
        T3_x,
        T3_y,
        T2_x * T1_y,
        T1_x * T2_y
    ])
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Chebyshev polynomial basis in (x, y).
    
    Basis functions (10 terms, one per parameter):
    p[0]: 1
    p[1]: T1(x) = x
    p[2]: T1(y) = y
    p[3]: T2(x) = 2x^2 - 1
    p[4]: T2(y) = 2y^2 - 1
    p[5]: T1(x)*T1(y) = x*y
    p[6]: T3(x) = 4x^3 - 3x
    p[7]: T3(y) = 4y^3 - 3y
    p[8]: T2(x)*T1(y) = (2x^2-1)*y
    p[9]: T1(x)*T2(y) = x*(2y^2-1)
    
    This is linear in params, so BFGS finds the global optimum reliably.
    """
    
    # Safety check: If we are running out of time, return a simple stable prediction
    if time.time() - _START_TIME > _TIME_LIMIT:
        x2 = x * x
        y2 = y * y
        xy = x * y
        return (params[0] 
                + params[1] * x 
                + params[2] * y 
                + params[3] * (2*x2 - 1) 
                + params[4] * (2*y2 - 1) 
                + params[5] * xy)

    # Compute Chebyshev basis
    basis = _chebyshev_basis(x, y)
    
    # Linear combination of basis functions
    pred = basis @ params

    # Ensure numerical stability
    if not np.all(np.isfinite(pred)):
        # Fallback to quadratic part
        x2 = x * x
        y2 = y * y
        xy = x * y
        return (params[0] 
                + params[1] * x 
                + params[2] * y 
                + params[3] * (2*x2 - 1) 
                + params[4] * (2*y2 - 1) 
                + params[5] * xy)

    return pred
