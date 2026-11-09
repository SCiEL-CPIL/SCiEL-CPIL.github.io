import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a spectral (sine/cosine) basis.
    This is better conditioned for BFGS than polynomial bases because:
    1. sin/cos have bounded derivatives (no ill-conditioning at large amplitudes)
    2. Oscillatory dynamics are naturally represented in frequency space
    3. Fewer effective degrees of freedom needed to capture periodic structure
    """
    # Scale x and v to approximately [-1, 1] based on training ranges
    # x: [-0.5728, 0.5275] -> mid -0.02265, half-range 0.55015
    # v: [-0.1754, 0.2821] -> mid 0.05335, half-range 0.22875
    
    x_mid = -0.02265
    x_scale = 0.55015
    v_mid = 0.05335
    v_scale = 0.22875
    
    xn = (x - x_mid) / x_scale
    vn = (v - v_mid) / v_scale
    
    # Clip to [-1, 1] to avoid numerical issues
    xn = np.clip(xn, -1.0, 1.0)
    vn = np.clip(vn, -1.0, 1.0)
    
    # Convert to angle in [0, 2*pi] for Fourier basis
    # Map [-1, 1] to [0, 2*pi]
    theta_x = np.pi * (xn + 1.0)  # [0, 2*pi]
    theta_v = np.pi * (vn + 1.0)  # [0, 2*pi]
    
    # Build 10 basis terms using sine and cosine
    # Order:
    # 0: 1 (constant)
    # 1: cos(theta_x)
    # 2: sin(theta_x)
    # 3: cos(theta_v)
    # 4: sin(theta_v)
    # 5: cos(2*theta_x)
    # 6: sin(2*theta_x)
    # 7: cos(theta_x)*cos(theta_v)
    # 8: sin(theta_x)*cos(theta_v)
    # 9: cos(theta_x)*sin(theta_v)
    
    b0 = np.ones_like(theta_x)
    b1 = np.cos(theta_x)
    b2 = np.sin(theta_x)
    b3 = np.cos(theta_v)
    b4 = np.sin(theta_v)
    b5 = np.cos(2.0 * theta_x)
    b6 = np.sin(2.0 * theta_x)
    b7 = b1 * b3
    b8 = b2 * b3
    b9 = b1 * b4
    
    return (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
