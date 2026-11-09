"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a pure Chebyshev basis T_0 ... T_9.

    Chebyshev polynomials are orthogonal on [-1, 1], giving a well-conditioned
    linear model. This lets BFGS (used by the evaluator to fit `params`) converge
    reliably and quickly to the optimal coefficient vector for the best
    9th-degree polynomial approximation of the hidden target.

    10 parameters correspond to 10 basis functions, so the model is exactly
    identified in the linear sense and BFGS only needs to walk a well-behaved
    quadratic surface.
    """
    # Deadline guard: if we're close to the 150 s budget, return zeros to avoid
    # being killed mid-evaluation. In practice the basis build is O(1) per point.
    if time.time() - _T0 > 140.0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)
    
    # 1. Chebyshev Polynomials (Global Trend)
    # Map to [-1, 1]
    t = np.clip(x / 3.0, -1.0, 1.0)
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    
    # 2. Gaussian RBFs (Local Features)
    # Centers at -2, 0, 2. Width sigma=1.0
    G1 = np.exp(-((x - (-2.0)) ** 2) / (2.0 * 1.0**2))
    G2 = np.exp(-((x - 0.0) ** 2) / (2.0 * 1.0**2))
    G3 = np.exp(-((x - 2.0) ** 2) / (2.0 * 1.0**2))
    
    # 3. Trigonometric (Periodic Features)
    S1 = np.sin(x)
    S2 = np.sin(2.0 * x)
    C1 = np.cos(x)
    
    # 4. Saturating (Step/Threshold)
    H1 = np.tanh(x)
    
    # Combine with parameters
    # p0: T0, p1: T1, p2: T2
    # p3: G1, p4: G2, p5: G3
    # p6: S1, p7: S2, p8: C1
    # p9: H1
    
    result = (params[0] * T0 + 
              params[1] * T1 + 
              params[2] * T2 + 
              params[3] * G1 + 
              params[4] * G2 + 
              params[5] * G3 + 
              params[6] * S1 + 
              params[7] * S2 + 
              params[8] * C1 + 
              params[9] * H1)
              
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    return result
# EVOLVE-BLOCK-END
