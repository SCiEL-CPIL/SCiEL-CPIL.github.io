"""Symbolic regression with 1 input(s) - Chebyshev + local Gaussian features."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill


def _chebyshev_basis(t: np.ndarray) -> np.ndarray:
    """Compute Chebyshev polynomials T_0 through T_6 for t in [-1, 1]."""
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    T6 = 32.0 * t**6 - 48.0 * t**4 + 18.0 * t**2 - 1.0
    return np.stack([T0, T1, T2, T3, T4, T5, T6], axis=-1)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Linear combination of 10 basis functions on [-2, 2].
    
    Combines the structural stability of a Chebyshev polynomial basis (7 terms,
    T_0 through T_6, well-conditioned on the bounded domain) with 3 localized
    Gaussian features that can capture local peaks/valleys that a pure
    polynomial cannot easily represent.
    
    Basis functions:
      0: T_0(x/2) = 1
      1: T_1(x/2) = x/2
      2: T_2(x/2) = 2*(x/2)^2 - 1
      3: T_3(x/2)
      4: T_4(x/2)
      5: T_5(x/2)
      6: T_6(x/2)
      7: exp(-x^2)              (Gaussian centered at 0)
      8: exp(-(x-1)^2)          (Gaussian at +1)
      9: exp(-(x+1)^2)          (Gaussian at -1)
    
    The Chebyshev part provides a stable global approximation, while the
    Gaussians add local flexibility for non-polynomial features.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev basis T_0 through T_6
    T0, T1, T2, T3, T4, T5, T6 = _chebyshev_basis(t)
    
    # Gaussian local features
    exp_arg1 = -x * x
    exp_arg1 = np.maximum(exp_arg1, -100.0)
    b7 = np.exp(exp_arg1)
    
    exp_arg2 = -(x - 1.0) * (x - 1.0)
    exp_arg2 = np.maximum(exp_arg2, -100.0)
    b8 = np.exp(exp_arg2)
    
    exp_arg3 = -(x + 1.0) * (x + 1.0)
    exp_arg3 = np.maximum(exp_arg3, -100.0)
    b9 = np.exp(exp_arg3)
    
    # Linear combination
    y = (params[0] * T0 + params[1] * T1 + params[2] * T2 +
         params[3] * T3 + params[4] * T4 + params[5] * T5 +
         params[6] * T6 +
         params[7] * b7 + params[8] * b8 + params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
