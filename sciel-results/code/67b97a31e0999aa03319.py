import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Chebyshev polynomial basis.
    Inputs are mapped to [-1, 1] to improve conditioning.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Map x to [-1, 1]
    # x range: [-0.5728, 0.5275]
    x_center = -0.02265
    x_scale = 0.55015
    X = (x - x_center) / x_scale
    
    # Map v to [-1, 1]
    # v range: [-0.1754, 0.2821]
    v_center = 0.05335
    v_scale = 0.22875
    V = (v - v_center) / v_scale
    
    # Chebyshev polynomials
    T0_X = 1.0
    T1_X = X
    T2_X = 2.0 * X**2 - 1.0
    T3_X = 4.0 * X**3 - 3.0 * X
    T4_X = 8.0 * X**4 - 8.0 * X**2 + 1.0
    
    T0_V = 1.0
    T1_V = V
    T2_V = 2.0 * V**2 - 1.0
    T3_V = 4.0 * V**3 - 3.0 * V
    T4_V = 8.0 * V**4 - 8.0 * V**2 + 1.0
    
    # Coupling term
    XV = X * V
    
    # Basis terms:
    # 0: T0(X) (Constant part of X)
    # 1: T1(X)
    # 2: T2(X)
    # 3: T3(X)
    # 4: T4(X)
    # 5: T1(V)
    # 6: T2(V)
    # 7: T3(V)
    # 8: T4(V)
    # 9: X*V (Coupling)
    
    return p0 * T0_X + p1 * T1_X + p2 * T2_X + p3 * T3_X + p4 * T4_X + \
           p5 * T1_V + p6 * T2_V + p7 * T3_V + p8 * T4_V + p9 * XV
