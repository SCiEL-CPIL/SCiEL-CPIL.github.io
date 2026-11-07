import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-inspired basis for a nonlinear oscillator (Duffing-like).
    Uses tightly scaled inputs to improve BFGS conditioning.
    Basis: polynomial terms up to cubic in x and v, plus physically-motivated cross terms.
    Avoids trig terms which are highly correlated with low-order polynomials in this range.
    """
    # Scale inputs to be roughly in [-1, 1] using tighter bounds
    # x in [-0.5728, 0.5275] -> use 0.55 as scale
    # v in [-0.1754, 0.2821] -> use 0.25 as scale
    X = x / 0.55
    V = v / 0.25
    
    # Polynomial terms
    X2 = X * X
    X3 = X2 * X
    
    V2 = V * V
    V3 = V2 * V
    
    # Cross terms
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Basis (10 terms, physically motivated, low correlation):
    # 0: 1
    # 1: X       (linear restoring force)
    # 2: X^2     (quadratic position)
    # 3: X^3     (cubic nonlinearity - Duffing)
    # 4: V       (linear damping)
    # 5: V^2     (quadratic velocity)
    # 6: V^3     (cubic damping)
    # 7: X*V     (cross term)
    # 8: X^2*V   (nonlinear damping)
    # 9: X*V^2   (another cross term)
    
    t0 = np.ones_like(x)
    t1 = X
    t2 = X2
    t3 = X3
    t4 = V
    t5 = V2
    t6 = V3
    t7 = XV
    t8 = X2V
    t9 = XV2
    
    T = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])
    
    return T @ params
