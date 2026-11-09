import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structural Change: Model time-varying coefficients (parametric excitation)
    # and include bias and cubic nonlinearity.
    # Basis:
    # 1-2: Fundamental drive (cos, sin)
    # 3-4: Position modulated by drive (parametric stiffness)
    # 5-6: Velocity modulated by drive (parametric damping)
    # 7-8: Linear state terms (constant stiffness/damping)
    # 9: Cubic nonlinearity (Duffing)
    # 10: Bias
    
    c1 = np.cos(t)
    s1 = np.sin(t)
    
    # Parametric terms
    xc1 = x * c1
    xs1 = x * s1
    vc1 = v * c1
    vs1 = v * s1
    
    # Linear state terms
    x1 = x
    v1 = v
    
    # Nonlinear restoring force
    x3 = x**3
    
    # Bias
    ones = np.ones_like(t)
    
    return (params[0] * c1 +
            params[1] * s1 +
            params[2] * xc1 +
            params[3] * xs1 +
            params[4] * vc1 +
            params[5] * vs1 +
            params[6] * x1 +
            params[7] * v1 +
            params[8] * x3 +
            params[9] * ones)
