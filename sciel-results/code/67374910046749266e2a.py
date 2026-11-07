import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Gaussian Process-inspired kernel basis
    # p0, p1: center of kernel 1 in (x, v)
    # p2: weight of kernel 1
    # p3, p4: center of kernel 2 in (x, v)
    # p5: weight of kernel 2
    # p6: shared inverse scale (squared for positivity)
    # p7: frequency of time oscillation
    # p8: amplitude of time oscillation
    # p9: bias
    
    cx1 = params[0]
    cv1 = params[1]
    w1 = params[2]
    
    cx2 = params[3]
    cv2 = params[4]
    w2 = params[5]
    
    inv_scale_sq = params[6] * params[6] + 1e-6  # Ensure positivity
    
    freq = params[7]
    amp = params[8]
    bias = params[9]
    
    # Gaussian kernels in (x, v) space
    dx1 = x - cx1
    dv1 = v - cv1
    r1_sq = dx1 * dx1 + dv1 * dv1
    kernel1 = w1 * np.exp(-r1_sq / (2.0 * inv_scale_sq))
    
    dx2 = x - cx2
    dv2 = v - cv2
    r2_sq = dx2 * dx2 + dv2 * dv2
    kernel2 = w2 * np.exp(-r2_sq / (2.0 * inv_scale_sq))
    
    # Time-dependent oscillation
    time_term = amp * np.sin(freq * t)
    
    result = kernel1 + kernel2 + time_term + bias
    
    return result
