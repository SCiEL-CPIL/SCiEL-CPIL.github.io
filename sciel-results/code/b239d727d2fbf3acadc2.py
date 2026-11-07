import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
        # u_s = s / (1.0 + s)
        
        # Temperature polynomial part, gated by substrate (cubic)
        t_poly = params[1] + params[2]*temp + params[3]*temp**2 + params[4]*temp**3
        y_temp = s * t_poly
        
        # pH part (cubic)
        y_pH = params[5]*pH + params[6]*pH**2 + params[7]*pH**3
        
        # Density part
        y_b = params[8]*b
        
        # Interactions
        y_s_T = params[9] * s * temp
        
        return y_temp + y_pH + y_b + y_s_T
