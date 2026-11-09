"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cheb_t0(x):
    """Chebyshev T0(x) = 1"""
    return np.ones_like(x)

def _cheb_t1(x):
    """Chebyshev T1(x) = x"""
    return x

def _cheb_t2(x):
    """Chebyshev T2(x) = 2x^2 - 1"""
    return 2*x**2 - 1

def _cheb_t3(x):
    """Chebyshev T3(x) = 4x^3 - 3x"""
    return 4*x**3 - 3*x

def _cheb_t4(x):
    """Chebyshev T4(x) = 8x^4 - 8x^2 + 1"""
    return 8*x**4 - 8*x**2 + 1


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using separable Chebyshev polynomial basis.
    
    params is an array of length 10. We use all 10 parameters as coefficients for:
    1. T0(x)*T0(y) = 1
    2. T1(x)*T0(y) = x
    3. T0(x)*T1(y) = y
    4. T2(x)*T0(y) = 2x^2-1
    5. T0(x)*T2(y) = 2y^2-1
    6. T1(x)*T1(y) = x*y
    7. T3(x)*T0(y) = 4x^3-3x
    8. T0(x)*T3(y) = 4y^3-3y
    9. T1(x)*T2(y) = x*(2y^2-1)
    10. T2(x)*T1(y) = (2x^2-1)*y
    
    This uses orthogonal Chebyshev polynomials which improve conditioning.
    """
    # Ensure x and y are 1D arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute Chebyshev basis in x and y
    T0x = _cheb_t0(x)
    T1x = _cheb_t1(x)
    T2x = _cheb_t2(x)
    T3x = _cheb_t3(x)
    
    T0y = _cheb_t0(y)
    T1y = _cheb_t1(y)
    T2y = _cheb_t2(y)
    T3y = _cheb_t3(y)
    
    # Separable product basis terms
    term0 = T0x * T0y   # 1
    term1 = T1x * T0y   # x
    term2 = T0x * T1y   # y
    term3 = T2x * T0y   # 2x^2-1
    term4 = T0x * T2y   # 2y^2-1
    term5 = T1x * T1y   # x*y
    term6 = T3x * T0y   # 4x^3-3x
    term7 = T0x * T3y   # 4y^3-3y
    term8 = T1x * T2y   # x*(2y^2-1)
    term9 = T2x * T1y   # (2x^2-1)*y
    
    # Linear combination
    return (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + 
            p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 + 
            p8 * term8 + p9 * term9)
# EVOLVE-BLOCK-END
