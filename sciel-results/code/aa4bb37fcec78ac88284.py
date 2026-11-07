import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed basis for nonlinear oscillator.
    
    This basis includes terms common in nonlinear oscillator models
    such as the Duffing oscillator and Van der Pol oscillator.
    By restricting the hypothesis space to these physically meaningful
    terms, we may achieve better generalization and optimization stability
    compared to a generic polynomial basis.
    
    Basis functions (10 total):
    1. 1
    2. x_s
    3. v_s
    4. x_s^3
    5. x_s^2 * v_s
    6. x_s * v_s^2
    7. v_s^3
    8. x_s^5
    9. x_s * v_s
    10. v_s^2
    
    Where x_s and v_s are scaled inputs.
    """
    # Scale inputs to approx [-1, 1] for numerical stability
    # x range approx [-0.57, 0.53]
    # v range approx [-0.17, 0.28]
    
    x_c = 0.0
    x_r = 0.57
    v_c = 0.05
    v_r = 0.25
    
    x_s = (x - x_c) / x_r
    v_s = (v - v_c) / v_r
    
    # Clip to prevent overflow in high-order terms if data is slightly out of range
    x_s = np.clip(x_s, -2.0, 2.0)
    v_s = np.clip(v_s, -2.0, 2.0)
    
    # Physics-informed basis terms
    b0 = np.ones_like(x)
    b1 = x_s
    b2 = v_s
    b3 = x_s * x_s * x_s
    b4 = x_s * x_s * v_s
    b5 = x_s * v_s * v_s
    b6 = v_s * v_s * v_s
    b7 = x_s * x_s * x_s * x_s * x_s
    b8 = x_s * v_s
    b9 = v_s * v_s
    
    # Linear combination
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
