import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process-inspired kernel basis for nonlinear oscillator.
    10 parameters define centers, scales, and weights for smooth Gaussian bumps
    in the (x, v) state space, plus a time-dependent drive.
    
    params layout:
      p[0]  = weight for bump 1
      p[1]  = center_x for bump 1
      p[2]  = center_v for bump 1
      p[3]  = weight for bump 2
      p[4]  = center_x for bump 2
      p[5]  = center_v for bump 2
      p[6]  = scale (shared bandwidth for both bumps)
      p[7]  = amplitude of cos(omega*t) drive
      p[8]  = frequency omega
      p[9]  = bias / offset
    """
    w1 = params[0]
    cx1 = params[1]
    cv1 = params[2]
    w2 = params[3]
    cx2 = params[4]
    cv2 = params[5]
    scale = params[6]
    drive_amp = params[7]
    omega = params[8]
    bias = params[9]
    
    # Ensure scale is positive and not too small to avoid numerical issues
    s = np.maximum(np.abs(scale), 0.05)
    
    # Gaussian bump 1 in (x, v) space
    d1 = (x - cx1) ** 2 + (v - cv1) ** 2
    bump1 = np.exp(-d1 / (2.0 * s * s))
    
    # Gaussian bump 2 in (x, v) space
    d2 = (x - cx2) ** 2 + (v - cv2) ** 2
    bump2 = np.exp(-d2 / (2.0 * s * s))
    
    # Time-dependent drive
    drive = drive_amp * np.cos(omega * t)
    
    result = w1 * bump1 + w2 * bump2 + drive + bias
    
    return result
