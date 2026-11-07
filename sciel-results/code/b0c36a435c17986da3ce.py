import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator.
    
    This model uses a hybrid basis that combines the structural physics of the
    Van der Pol oscillator with the numerical stability of a Chebyshev-like
    orthogonal expansion.
    
    Key structural choices:
    1. Input scaling to [-1, 1] to improve conditioning for BFGS.
    2. Explicit Van der Pol term: x^2 * v (energy injection/dissipation).
    3. Duffing term: x^3 (cubic stiffness).
    4. Higher-order flexibility terms (x^4, v^3, x^2) to capture asymmetries 
       and higher-order dynamics not captured by the simplest oscillator models.
    5. Avoids pure high-degree polynomials in one variable to reduce 
       collinearity, using mixed terms and bounded scaling instead.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Scale inputs to [-1, 1] based on training ranges
    # x range: [-0.5728, 0.5275] -> center = -0.02265, half_range = 0.55015
    # v range: [-0.1754, 0.2821] -> center = 0.05335, half_range = 0.22875
    
    x_center = -0.02265
    x_scale = 0.55015
    v_center = 0.05335
    v_scale = 0.22875
    
    X = (x - x_center) / x_scale
    V = (v - v_center) / v_scale
    
    # Clip to ensure stability if test data slightly exceeds training range
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # 2. Pre-compute monomials
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    
    V2 = V * V
    V3 = V2 * V
    
    # Key coupling terms
    XV = X * V
    X2V = X2 * V
    
    # 3. Construct Basis Terms
    # We select 10 terms that prioritize physical oscillator dynamics:
    
    # Term 0: Constant (offset)
    b0 = 1.0
    
    # Term 1: Linear X (Linear restoring force)
    b1 = X
    
    # Term 2: Linear V (Linear damping)
    b2 = V
    
    # Term 3: Cubic X (Duffing nonlinearity - hard/soft spring)
    b3 = X3
    
    # Term 4: Quadratic V (Nonlinear damping component)
    b4 = V2
    
    # Term 5: Cross term X*V (Coupling)
    b5 = XV
    
    # Term 6: Quintic X (Higher order stiffness, captures saturation)
    # Note: X is in [-1,1], so X^5 is well-behaved.
    X5 = X4 * X
    b6 = X5
    
    # Term 7: Cubic V (Higher order damping)
    b7 = V3
    
    # Term 8: Van der Pol Term X^2 * V (Amplitude-dependent damping)
    # This is the critical term for limit cycle behavior.
    b8 = X2V
    
    # Term 9: Quartic X (Even nonlinearity, asymmetry)
    b9 = X4
    
    # 4. Linear Combination
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + \
           p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9
