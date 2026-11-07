import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using a mixed Chebyshev-monomial basis.
    
    Combines the orthogonality benefits of Chebyshev polynomials (inspiration 1)
    with physics-informed monomial terms (current program) for better conditioning
    and expressiveness.
    
    Basis terms:
    - Chebyshev polynomials T1-X, T2-X, T3-X for position nonlinearity
    - Chebyshev polynomials T1-V, T2-V for velocity nonlinearity
    - Monomial cross terms: X*V, X^2*V (Van der Pol damping)
    - Higher-order monomials: X^3, V^3 for strong nonlinearities
    """
    # Normalize inputs to approximately [-1, 1]
    # Training ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # Use center and half-width for better centering
    x_center = -0.02265
    x_scale = 0.55015
    v_center = 0.05335
    v_scale = 0.22875
    
    X = (x - x_center) / x_scale
    V = (v - v_center) / v_scale
    
    # Clip to [-1, 1] to stay within Chebyshev domain (training range)
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Precompute powers for efficiency
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    # Chebyshev polynomials of first kind
    # T0 = 1, T1 = x, T2 = 2x^2 - 1, T3 = 4x^3 - 3x
    T1_X = X
    T2_X = 2.0 * X2 - 1.0
    T3_X = 4.0 * X3 - 3.0 * X
    
    T1_V = V
    T2_V = 2.0 * V2 - 1.0
    
    # Basis terms:
    # 0: Constant (driving force)
    t0 = 1.0
    # 1: Linear restoring force (T1_X)
    t1 = T1_X
    # 2: Quadratic position (T2_X)
    t2 = T2_X
    # 3: Cubic restoring force (T3_X)
    t3 = T3_X
    # 4: Linear damping (T1_V)
    t4 = T1_V
    # 5: Quadratic velocity (T2_V)
    t5 = T2_V
    # 6: Cross term X*V (linear coupling)
    t6 = X * V
    # 7: Amplitude-dependent damping (X^2 * V, Van der Pol)
    t7 = X2 * V
    # 8: Cubic damping (V^3)
    t8 = V3
    # 9: Cubic position monomial (X^3) for strong nonlinearity
    t9 = X3
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Combine terms
    return (p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3 +
            p4 * t4 + p5 * t5 + p6 * t6 + p7 * t7 +
            p8 * t8 + p9 * t9)
