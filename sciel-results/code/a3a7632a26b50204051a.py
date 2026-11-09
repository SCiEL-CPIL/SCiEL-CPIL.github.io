"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis model for symbolic regression with 1 input.
    
    Combines the structural idea of a well-conditioned orthogonal polynomial basis
    (Legendre polynomials P_0..P_4, giving 5 parameters) with a set of non-polynomial
    basis functions (sin, cos, tanh, exp, |x|) to capture potential non-polynomial
    features of the hidden function.
    
    The Legendre basis is orthogonal on [-1,1] (after scaling x from [-3,3]), which
    keeps the design matrix well-conditioned for BFGS convergence. The additional
    non-polynomial basis functions provide flexibility to capture oscillatory,
    saturating, or exponential behavior that a pure polynomial of degree 9 cannot.
    
    Basis functions (10 total):
    1. P_0(u) = 1                    (constant)
    2. P_1(u) = u                    (linear)
    3. P_2(u) = (3u^2 - 1)/2         (quadratic)
    4. P_3(u) = (5u^3 - 3u)/2        (cubic)
    5. P_4(u) = (35u^4 - 30u^2 + 3)/8 (quartic)
    6. sin(x)
    7. cos(x)
    8. tanh(x)
    9. exp(x/3)
    10. |x|
    
    This hybrid approach aims to improve upon the current program's mixed basis
    (which includes higher-order powers x^2, x^3 that are less well-conditioned)
    by replacing them with orthogonal Legendre polynomials, while retaining the
    non-polynomial features that may be important for the hidden function.
    """
    _T0 = time.time()
    _DEADLINE = 140.0  # Safety margin before 150s wall-clock limit
    
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is 1D for consistency
    if x.ndim > 1:
        x = x.ravel()
    
    n = x.size
    if n == 0:
        return np.array([])
    
    # Scale input to [-1, 1] for Legendre polynomials
    u = x / 3.0
    
    # Compute Legendre polynomials P_0 through P_4 using recurrence
    # P_0(u) = 1
    # P_1(u) = u
    # (n+1) P_{n+1}(u) = (2n+1) u P_n(u) - n P_{n-1}(u)
    
    P0 = np.ones_like(u)
    P1 = u.copy()
    
    # P_2(u)
    P2 = (3.0 * u * P1 - 1.0 * P0) / 2.0
    
    # P_3(u)
    P3 = (5.0 * u * P2 - 2.0 * P1) / 3.0
    
    # P_4(u)
    P4 = (7.0 * u * P3 - 3.0 * P2) / 4.0
    
    # Non-polynomial basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    tanh_x = np.tanh(x)
    exp_x = np.exp(x / 3.0)  # Scaled to avoid overflow for x in [-3,3]
    abs_x = np.abs(x)
    
    # Stack all 10 basis functions
    basis = np.column_stack([
        P0,       # 1: constant
        P1,       # 2: linear
        P2,       # 3: quadratic (Legendre)
        P3,       # 4: cubic (Legendre)
        P4,       # 5: quartic (Legendre)
        sin_x,    # 6
        cos_x,    # 7
        tanh_x,   # 8
        exp_x,    # 9
        abs_x,    # 10
    ])
    
    # Linear combination: params @ basis.T
    result = basis @ params
    
    # Finite-output safety net
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    # Deadline guard
    if time.time() - _T0 > _DEADLINE:
        return result
    
    return result
# EVOLVE-BLOCK-END
