# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Hybrid Fourier-MLP model.
    - 5 params for a global Fourier basis: [1, sin(x), cos(x), sin(y), cos(y)]
    - 5 params for a single hidden neuron MLP: tanh(s * (w_x*x + w_y*y + b)) * w_out
    
    This combines global periodic trends (Fourier) with local non-linear flexibility (MLP).
    The MLP uses a learnable scale factor 's' to adjust the steepness of the tanh activation,
    allowing it to better fit sharp transitions or localized features.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Global Fourier Basis
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    
    # 5 Fourier coefficients: constant, sin(x), cos(x), sin(y), cos(y)
    fourier_part = (p[0] + 
                    p[1] * sin_x + 
                    p[2] * cos_x + 
                    p[3] * sin_y + 
                    p[4] * cos_y)
    
    # Local Non-linear MLP Component (Single Hidden Neuron)
    # h = tanh(s * (w_x * x + w_y * y + b))
    # out_mlp = w_out * h
    s = p[5]
    w_x = p[6]
    w_y = p[7]
    b = p[8]
    w_out = p[9]
    
    hidden = np.tanh(s * (w_x * x + w_y * y + b))
    mlp_part = w_out * hidden
    
    return fourier_part + mlp_part
# EVOLVE-BLOCK-END
