import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration of a nonlinear oscillator using a physically-motivated basis
    with both polynomial and saturated (tanh) nonlinearities.
    
    The model structure is:
    a = p0 + p1*xn + p2*xn^3 + p3*tanh(xn) + p4*vn + p5*vn^3 + p6*tanh(vn) + p7*xn*vn + p8*xn^2*vn + p9*vn^2*xn
    
    where xn and vn are normalized inputs.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [-1, 1] based on training data ranges
    # x range: [-0.5728, 0.5275] -> mid: -0.02265, half_range: 0.55015
    # v range: [-0.1754, 0.2821] -> mid: 0.05335, half_range: 0.22875
    
    x_mid = -0.02265
    x_half = 0.55015
    v_mid = 0.05335
    v_half = 0.22875
    
    # Avoid division by zero, though ranges are fixed
    xn = (x - x_mid) / x_half
    vn = (v - v_mid) / v_half
    
    # Precompute basis functions
    # Linear and Cubic terms
    x_lin = xn
    x_cub = xn * xn * xn
    v_lin = vn
    v_cub = vn * vn * vn
    
    # Saturated terms
    x_sat = np.tanh(xn)
    v_sat = np.tanh(vn)
    
    # Interaction terms
    x_v = xn * vn
    x2_v = xn * xn * vn
    v2_x = vn * vn * xn
    
    # Construct the model
    # p0: bias
    # p1: linear stiffness
    # p2: cubic stiffness (Duffing)
    # p3: saturated stiffness
    # p4: linear damping
    # p5: cubic damping
    # p6: saturated damping
    # p7: linear cross-coupling
    # p8: nonlinear drag (x^2 v)
    # p9: nonlinear modulation (v^2 x)
    
    return (p0 + 
            p1 * x_lin + 
            p2 * x_cub + 
            p3 * x_sat + 
            p4 * v_lin + 
            p5 * v_cub + 
            p6 * v_sat + 
            p7 * x_v + 
            p8 * x2_v + 
            p9 * v2_x)
