import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
        u_s = s / (1.0 + s)
        
        # Temperature polynomial part, gated by substrate (quadratic)
        t_poly = params[1] + params[2]*temp + params[3]*temp**2
        y_temp = u_s * t_poly
        
        # pH part
        y_pH = params[4]*pH + params[5]*pH**2
        
        # Density part
        y_b = params[6]*b
        
        # Interactions
        y_s_pH = params[7] * u_s * pH
        y_T_pH = params[8] * temp * pH
        y_s_T = params[9] * u_s * temp
        
        return params[0] + y_temp + y_pH + y_b + y_s_pH + y_T_pH + y_s_T
