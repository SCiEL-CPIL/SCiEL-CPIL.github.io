import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Monod-like substrate saturation with fitted half-saturation
    # params[0] controls the saturation constant K_s
    K_s = 1.0 + np.abs(params[0])
    u_s = s / (K_s + s)
    
    # Temperature: cubic polynomial (captures non-monotonic response)
    # Gated by substrate to reflect that growth requires both substrate and suitable T
    t_poly = params[1] + params[2]*temp + params[3]*temp**2 + params[4]*temp**3
    y_temp = u_s * t_poly
    
    # pH: quadratic polynomial (captures optimal pH region)
    y_pH = params[5] + params[6]*pH + params[7]*pH**2
    
    # Interactions:
    # Substrate x pH interaction
    y_s_pH = params[8] * u_s * pH
    
    # Temperature x pH interaction
    y_T_pH = params[9] * temp * pH
    
    # Note: density (b) is not explicitly modeled with a separate coefficient here.
    # The 10 parameters are allocated to:
    # 0: K_s adjustment
    # 1-4: temp polynomial
    # 5-7: pH polynomial
    # 8: s-pH interaction
    # 9: T-pH interaction
    
    # If density has a strong effect, it may be captured indirectly through the 
    # other terms or the data may show weak density dependence in the training range.
    # Alternatively, we could drop one of the interaction terms to include density.
    # Given the current best score, let's try including density by dropping the T-pH interaction
    # and using a linear density term.
    
    # Revised allocation:
    # 0: K_s adjustment
    # 1-4: temp polynomial (cubic)
    # 5-6: pH polynomial (quadratic)
    # 7: density linear
    # 8: s-pH interaction
    # 9: T-pH interaction
    
    # Let's recompute with this allocation
    y_b = params[7] * b
    
    return params[0]*0.0 + y_temp + y_pH + y_b + y_s_pH + y_T_pH
