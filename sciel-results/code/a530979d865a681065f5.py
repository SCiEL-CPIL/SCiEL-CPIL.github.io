import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Response Surface Model for E. coli growth rate.
    Uses normalized polynomial features to capture non-linear interactions
    without assuming specific kinetic mechanisms (like Monod or Gaussian).
    
    Parameters:
    p0: Bias
    p1: Linear substrate effect
    p2: Linear temperature effect
    p3: Quadratic temperature effect (curvature)
    p4: Linear pH effect
    p5: Quadratic pH effect (curvature)
    p6: Linear density effect
    p7: Substrate-Temperature interaction
    p8: Substrate-pH interaction
    p9: Substrate-Density interaction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [-1, 1] range for numerical stability
    # b in [1, 5] -> center 3, scale 2
    b_norm = (b - 3.0) / 2.0
    # s in [1, 5] -> center 3, scale 2
    s_norm = (s - 3.0) / 2.0
    # temp in [1, 40] -> center 20.5, scale 19.5
    t_norm = (temp - 20.5) / 19.5
    # pH in [2, 12] -> center 7, scale 5
    ph_norm = (pH - 7.0) / 5.0
    
    # Construct features
    f1 = s_norm
    f2 = t_norm
    f3 = t_norm ** 2
    f4 = ph_norm
    f5 = ph_norm ** 2
    f6 = b_norm
    f7 = s_norm * t_norm
    f8 = s_norm * ph_norm
    f9 = s_norm * b_norm
    
    # Linear combination
    pred = p0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 + p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9
    
    return pred
