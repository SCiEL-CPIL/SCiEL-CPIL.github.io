import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Scale inputs to O(1)
        t_s = (t - 40.0) / 10.0
        x_s = x / 0.3
        v_s = v / 0.2
        
        # Basis functions
        f0 = np.ones_like(x_s)
        f1 = t_s
        f2 = x_s
        f3 = v_s
        f4 = x_s**2
        f5 = x_s**3
        f6 = v_s**2
        f7 = x_s * v_s
        f8 = x_s**2 * v_s
        f9 = x_s * v_s**2
        
        return (params[0] * f0 + params[1] * f1 + params[2] * f2 + params[3] * f3 +
                params[4] * f4 + params[5] * f5 + params[6] * f6 + params[7] * f7 +
                params[8] * f8 + params[9] * f9)
