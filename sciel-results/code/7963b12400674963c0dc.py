# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using physically-motivated basis:
    Duffing-Van der Pol structure with periodic drive.
    
    Basis functions:
    0: 1 (bias)
    1: sin(t) (periodic drive, sine component)
    2: cos(t) (periodic drive, cosine component)
    3: x (linear spring)
    4: x^3 (Duffing cubic nonlinearity)
    5: v (linear damping)
    6: v * x^2 (state-dependent damping, Van der Pol style)
    7: x * v (cross term)
    8: x^2 * v (another cross term)
    9: sin(2t) (second harmonic of drive)
    
    This structure is motivated by the canonical Duffing-Van der Pol oscillator:
    dv/dt = -mu*v*(1-x^2) - x + beta*x^3 + F*sin(t)
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Periodic drive terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    sin_2t = np.sin(2.0 * t)
    
    # State-dependent terms
    x2 = x * x
    x3 = x2 * x
    
    # Damping terms (Van der Pol style)
    v_damp = v * (1.0 - x2)  # Van der Pol damping
    
    # Cross terms
    xv = x * v
    x2v = x2 * v
    
    # 10 basis functions
    b0 = np.ones_like(x)
    b1 = sin_t
    b2 = cos_t
    b3 = x
    b4 = x3
    b5 = v
    b6 = v_damp
    b7 = xv
    b8 = x2v
    b9 = sin_2t
    
    result = (p[0] * b0 +
              p[1] * b1 +
              p[2] * b2 +
              p[3] * b3 +
              p[4] * b4 +
              p[5] * b5 +
              p[6] * b6 +
              p[7] * b7 +
              p[8] * b8 +
              p[9] * b9)
    
    # Safety: replace any non-finite values with 0
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
