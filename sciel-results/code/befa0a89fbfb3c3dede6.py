import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a scaled Chebyshev basis.
    This improves numerical conditioning for the BFGS optimizer compared to raw polynomials.
    The basis includes terms corresponding to the Duffing oscillator dynamics 
    (linear restoring, cubic nonlinearity, linear damping) and additional flexible terms.
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
    
    # Clip to [-1, 1] to avoid numerical issues if test data is slightly out of range
    # though the problem states "Inputs in training range"
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # 2. Compute Chebyshev polynomials
    # T0 = 1
    # T1 = X
    # T2 = 2X^2 - 1
    # T3 = 4X^3 - 3X
    
    X2 = X * X
    X3 = X2 * X
    T1_X = X
    T2_X = 2.0 * X2 - 1.0
    T3_X = 4.0 * X3 - 3.0 * X
    
    V2 = V * V
    V3 = V2 * V
    T1_V = V
    T2_V = 2.0 * V2 - 1.0
    T3_V = 4.0 * V3 - 3.0 * V
    
    # 3. Construct Basis Terms
    # We select 10 terms that cover the physical dynamics and add flexibility.
    
    # Term 0: Constant
    b0 = 1.0
    
    # Term 1: Linear X (Restoring force)
    b1 = T1_X
    
    # Term 2: Cubic X (Duffing nonlinearity)
    b2 = T3_X
    
    # Term 3: Linear V (Damping)
    b3 = T1_V
    
    # Term 4: Cubic V (Nonlinear damping)
    b4 = T3_V
    
    # Term 5: Cross term X*V (Coupling)
    b5 = X * V
    
    # Term 6: Cross term X^2 * V
    b6 = X2 * V
    
    # Term 7: Cross term X * V^2
    b7 = X * V2
    
    # Term 8: Quadratic X (Asymmetric restoring)
    b8 = T2_X
    
    # Term 9: Quadratic V (Asymmetric damping)
    b9 = T2_V
    
    # 4. Linear Combination
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9
