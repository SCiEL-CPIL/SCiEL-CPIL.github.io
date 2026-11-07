import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using Chebyshev basis.
    Chebyshev polynomials are orthogonal on [-1, 1], providing excellent
    numerical conditioning for BFGS optimization.
    """
    # Normalize inputs to approximately [-1, 1]
    X_SCALE = 0.5728
    V_SCALE = 0.2821
    
    X = x / X_SCALE
    V = v / V_SCALE
    
    # Clip to [-1, 1] to stay within Chebyshev domain (training range)
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Chebyshev polynomials of first kind: T_0=1, T_1=x, T_{n+1}=2x*T_n - T_{n-1}
    # For X:
    T0_X = np.ones_like(X)
    T1_X = X
    T2_X = 2.0 * X * T1_X - T0_X
    T3_X = 2.0 * X * T2_X - T1_X
    
    # For V:
    T0_V = np.ones_like(V)
    T1_V = V
    T2_V = 2.0 * V * T1_V - T0_V
    T3_V = 2.0 * V * T2_V - T1_V
    
    # Bivariate basis: tensor products of Chebyshev polynomials
    # Total degree <= 3, plus selected interaction terms
    t1 = T0_X * T0_V          # 1
    t2 = T1_X * T0_V          # T1(X)
    t3 = T2_X * T0_V          # T2(X)
    t4 = T3_X * T0_V          # T3(X)
    t5 = T0_X * T1_V          # T1(V)
    t6 = T0_X * T2_V          # T2(V)
    t7 = T0_X * T3_V          # T3(V)
    t8 = T1_X * T1_V          # T1(X)*T1(V)
    t9 = T2_X * T1_V          # T2(X)*T1(V)
    t10 = T1_X * T2_V         # T1(X)*T2(V)
    
    return (p0 * t1 + p1 * t2 + p2 * t3 + p3 * t4 + 
            p4 * t5 + p5 * t6 + p6 * t7 + p7 * t8 + 
            p8 * t9 + p9 * t10)
