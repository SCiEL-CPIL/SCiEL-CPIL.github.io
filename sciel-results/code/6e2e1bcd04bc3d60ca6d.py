# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 125.0  # safety margin before 150s kill

def _chebyshev_terms(z, degree=3):
    """Compute Chebyshev polynomial values T_0(z), T_1(z), ..., T_degree(z)."""
    n = len(z)
    T = np.zeros((n, degree + 1), dtype=np.float64)
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = z
    for k in range(2, degree + 1):
        T[:, k] = 2.0 * z * T[:, k-1] - T[:, k-2]
    return T

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev-basis nonlinear oscillator model with higher-order cross terms.
    
    Uses Chebyshev polynomials in normalized (x, v) coordinates up to degree 3
    for better numerical conditioning, combined with sinusoidal drive terms.
    
    Basis (10 terms):
    1.  T_0(x) * T_0(v) = 1 (bias)
    2.  sin(t) (drive)
    3.  cos(t) (drive)
    4.  T_1(x) * T_1(v) = x_norm * v_norm (bilinear coupling)
    5.  T_2(x) * T_0(v) = T_2(x_norm) (quadratic in x)
    6.  T_0(x) * T_2(v) = T_2(v_norm) (quadratic in v)
    7.  T_3(x) * T_0(v) = T_3(x_norm) (cubic in x, Duffing-like)
    8.  T_0(x) * T_3(v) = T_3(v_norm) (cubic in v, nonlinear damping)
    9.  T_2(x) * T_1(v) (cross quadratic-linear)
    10. T_1(x) * T_2(v) (cross linear-quadratic)
    
    Normalization: x_norm = (x - x_center) / x_scale, v_norm = (v - v_center) / v_scale
    Using training ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    n = len(t)
    
    # Normalize x and v to [-1, 1] range using known training bounds
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    x_center = (x_min + x_max) / 2.0
    x_scale = (x_max - x_min) / 2.0
    v_center = (v_min + v_max) / 2.0
    v_scale = (v_max - v_min) / 2.0
    
    x_norm = (x - x_center) / x_scale
    v_norm = (v - v_center) / v_scale
    
    # Clip to [-1.5, 1.5] for safety (slightly beyond [-1,1])
    x_norm = np.clip(x_norm, -1.5, 1.5)
    v_norm = np.clip(v_norm, -1.5, 1.5)
    
    # Compute Chebyshev terms up to degree 3
    Tx = _chebyshev_terms(x_norm, degree=3)  # shape (n, 4): T0, T1, T2, T3
    Tv = _chebyshev_terms(v_norm, degree=3)  # shape (n, 4): T0, T1, T2, T3
    
    # Build 10 basis functions
    b1 = np.ones(n)                          # T0(x)*T0(v)
    b2 = np.sin(t)                           # drive
    b3 = np.cos(t)                           # drive
    b4 = Tx[:, 1] * Tv[:, 1]                 # T1(x)*T1(v)
    b5 = Tx[:, 2] * Tv[:, 0]                 # T2(x)*T0(v)
    b6 = Tx[:, 0] * Tv[:, 2]                 # T0(x)*T2(v)
    b7 = Tx[:, 3] * Tv[:, 0]                 # T3(x)*T0(v)
    b8 = Tx[:, 0] * Tv[:, 3]                 # T0(x)*T3(v)
    b9 = Tx[:, 2] * Tv[:, 1]                 # T2(x)*T1(v)
    b10 = Tx[:, 1] * Tv[:, 2]                # T1(x)*T2(v)
    
    # Linear combination
    pred = (b1 * params[0] +
            b2 * params[1] +
            b3 * params[2] +
            b4 * params[3] +
            b5 * params[4] +
            b6 * params[5] +
            b7 * params[6] +
            b8 * params[7] +
            b9 * params[8] +
            b10 * params[9])
    
    # Safety: replace any non-finite values with 0
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
