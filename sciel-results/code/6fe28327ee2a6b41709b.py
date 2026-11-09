"""Symbolic regression with 3 inputs using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_coeffs(n: int) -> np.ndarray:
    """Return Chebyshev T_k coefficients for k=0..n-1 as a 1D array.
    
    We map input t in [0.5, 2] to s in [-1, 1] via s = (t - 1.25) / 0.75
    so that t=0.5 -> s=-1, t=2 -> s=1.
    
    Then T_k(s) is computed.
    """
    # We'll compute Chebyshev polynomials T_0, T_1, ..., T_{n-1}
    # T_0(s) = 1
    # T_1(s) = s
    # T_{k}(s) = 2*s*T_{k-1}(s) - T_{k-2}(s)
    # But we need to express in terms of t directly for efficient evaluation.
    # Actually, let's just compute the basis functions directly.
    pass


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Chebyshev polynomial basis mapped to [0.5, 2].
    
    We use 10 parameters total. The basis consists of products of Chebyshev
    polynomials in each variable, truncated to keep total degree <= 2.
    
    Mapping: s_x = (x - 1.25) / 0.75, similarly for y, z.
    
    Basis functions (total degree <= 2):
    Degree 0: T_0(sx)*T_0(sy)*T_0(sz) = 1
    Degree 1: T_1(sx)*T_0(sy)*T_0(sz) = sx
              T_0(sx)*T_1(sy)*T_0(sz) = sy
              T_0(sx)*T_0(sy)*T_1(sz) = sz
    Degree 2: T_2(sx)*T_0(sy)*T_0(sz) = T2(sx)
              T_0(sx)*T_2(sy)*T_0(sz) = T2(sy)
              T_0(sx)*T_0(sy)*T_2(sz) = T2(sz)
              T_1(sx)*T_1(sy)*T_0(sz) = sx*sy
              T_1(sx)*T_0(sy)*T_1(sz) = sx*sz
              T_0(sx)*T_1(sy)*T_1(sz) = sy*sz
    
    That's 1 + 3 + 6 = 10 basis functions, matching 10 params.
    """
    p = params
    
    # Map to [-1, 1]
    sx = (x - 1.25) / 0.75
    sy = (y - 1.25) / 0.75
    sz = (z - 1.25) / 0.75
    
    # Chebyshev T_0 = 1, T_1 = s, T_2 = 2*s^2 - 1
    t2x = 2.0 * sx * sx - 1.0
    t2y = 2.0 * sy * sy - 1.0
    t2z = 2.0 * sz * sz - 1.0
    
    # Build the 10 basis functions
    # 0: 1
    # 1: sx
    # 2: sy
    # 3: sz
    # 4: T2(sx)
    # 5: T2(sy)
    # 6: T2(sz)
    # 7: sx*sy
    # 8: sx*sz
    # 9: sy*sz
    
    result = (p[0]
              + p[1] * sx
              + p[2] * sy
              + p[3] * sz
              + p[4] * t2x
              + p[5] * t2y
              + p[6] * t2z
              + p[7] * sx * sy
              + p[8] * sx * sz
              + p[9] * sy * sz)
    
    return result
# EVOLVE-BLOCK-END
