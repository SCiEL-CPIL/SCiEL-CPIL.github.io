import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Nonlinear Oscillator Model: Autonomous Duffing/Van der Pol core 
    with Parametric and Direct Excitation terms.
    
    This model combines the structural stability of autonomous nonlinear 
    oscillator dynamics (Duffing restoring force, Van der Pol damping) with 
    explicit time-dependent driving terms (sin/cos) that were found to 
    improve fit quality in related candidates.
    
    Basis Functions:
      0: 1              (Offset)
      1: xn             (Linear restoring)
      2: xn^3           (Duffing cubic restoring)
      3: vn             (Linear damping)
      4: xn^2 * vn      (Van der Pol nonlinear damping)
      5: xn * vn        (State-dependent damping)
      6: xn * sin(t)    (Parametric excitation)
      7: sin(t)         (Direct sinusoidal drive)
      8: cos(t)         (Quadrature drive)
      9: vn^3           (Cubic velocity saturation)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalization for better conditioning
    # x range approx [-0.2056, 0.1749] -> scale 0.2
    # v range approx [-0.3391, 0.3624] -> scale 0.35
    xn = x / 0.2
    vn = v / 0.35

    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn

    # Trig terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    # Basis functions
    b0 = 1.0
    b1 = xn
    b2 = xn3
    b3 = vn
    b4 = xn2 * vn
    b5 = xn * vn
    b6 = xn * sin_t
    b7 = sin_t
    b8 = cos_t
    b9 = vn3

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
