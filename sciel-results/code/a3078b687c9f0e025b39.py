import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to approximately [-1, 1]
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2, 0.17] -> center -0.015, scale 0.2
    # v: [-0.34, 0.36] -> center 0.01, scale 0.35
    
    t_norm = (t - 40.0) / 10.0
    x_norm = (x + 0.015) / 0.2
    v_norm = (v - 0.01) / 0.35
    
    # Single hidden unit with tanh activation
    # params[0:4] are weights and bias for the hidden unit
    z = np.tanh(params[0] * t_norm + params[1] * x_norm + params[2] * v_norm + params[3])
    
    # Output is a polynomial of z
    # params[4:10] are coefficients
    # z^0 to z^5
    return (params[4] + 
            params[5] * z + 
            params[6] * z**2 + 
            params[7] * z**3 + 
            params[8] * z**4 + 
            params[9] * z**5)
