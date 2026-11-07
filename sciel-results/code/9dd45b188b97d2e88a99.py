import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    c_r1 = params[0]
    c_r2 = params[1]
    c_r3 = params[2]
    c_d1 = params[3]
    c_d2 = params[4]
    c_d3 = params[5]
    c_dr1 = params[6]
    c_dr2 = params[7]
    omega = params[8]
    c_bias = params[9]
    
    # Rational restoring force
    num_r = c_r1 * x + c_r2 * (x ** 3)
    den_r = 1.0 + c_r3 * (x ** 2)
    # Safety guard for denominator
    den_r = np.where(np.abs(den_r) < 1e-6, np.sign(den_r) * 1e-6 + 1e-6, den_r)
    restoring = num_r / den_r
    
    # Rational damping force
    num_d = c_d1 * v + c_d2 * (v ** 3)
    den_d = 1.0 + c_d3 * (v ** 2)
    den_d = np.where(np.abs(den_d) < 1e-6, np.sign(den_d) * 1e-6 + 1e-6, den_d)
    damping = num_d / den_d
    
    # Driving force
    drive = c_dr1 * np.cos(omega * t) + c_dr2 * np.sin(omega * t)
    
    result = restoring + damping + drive + c_bias
    
    return result
