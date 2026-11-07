"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    
    Model:
      a(t, x, v) = sum_{i=0}^{9} p_i * basis_i(t, x, v)
    
    This model is linear in the parameters, which allows the BFGS optimizer
    (or effectively a least-squares solver) to find the global minimum for
    the given basis functions quickly and robustly.
    
    Basis functions chosen to cover common nonlinear oscillator dynamics:
      0: 1                          (Offset / Mean)
      1: x                          (Linear restoring force)
      2: x^3                        (Cubic / Duffing restoring)
      3: v                          (Linear damping)
      4: x^2 * v                    (Van der Pol nonlinear damping)
      5: t                          (Slow time drift / linear phase)
      6: sin(t)                     (Sinusoidal drive, in-phase)
      7: cos(t)                     (Sinusoidal drive, quadrature)
      8: x * v                      (Parametric / gyroscopic coupling)
      9: v^3                        (Cubic velocity saturation)
    
    Inputs are normalized to improve conditioning:
      tn = t / 50.0       (Scale down time)
      xn = x / 0.2        (Scale position to ~[-1, 1])
      vn = v / 0.35       (Scale velocity to ~[-1, 1])
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalization to unit scale
    xn = x / 0.2
    vn = v / 0.35

    # Precompute basis functions (Autonomous dynamics focus)
    # 0: Offset
    b0 = 1.0
    # 1: Linear restoring
    b1 = xn
    # 2: Cubic restoring (Duffing)
    b2 = xn * xn * xn
    # 3: Linear damping
    b3 = vn
    # 4: Quadratic position * velocity (Van der Pol)
    b4 = (xn * xn) * vn
    # 5: Cubic velocity (Saturation)
    b5 = vn * vn * vn
    # 6: Position * Velocity (Gyroscopic/Parametric)
    b6 = xn * vn
    # 7: Position^2 * Velocity^2 (Energy coupling)
    b7 = (xn * xn) * (vn * vn)
    # 8: Position^4 (Higher order Duffing)
    b8 = xn * xn * xn * xn
    # 9: Position * Velocity^2 (Velocity squared coupling)
    b9 = xn * (vn * vn)

    # Linear combination
    return (p0 * b0 + 
            p1 * b1 + 
            p2 * b2 + 
            p3 * b3 + 
            p4 * b4 + 
            p5 * b5 + 
            p6 * b6 + 
            p7 * b7 + 
            p8 * b8 + 
            p9 * b9)
# EVOLVE-BLOCK-END
