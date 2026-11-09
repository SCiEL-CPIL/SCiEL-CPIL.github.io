import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = x.flatten()
    y = y.flatten()
    
    # Extract parameters
    # 0: linear coeff for x
    # 1: linear coeff for y
    # 2: weight for RBF 1
    # 3: log-width for RBF 1
    # 4: weight for RBF 2
    # 5: log-width for RBF 2
    # 6: weight for RBF 3
    # 7: log-width for RBF 3
    # 8: weight for RBF 4
    # 9: log-width for RBF 4
    
    cx = params[0]
    cy = params[1]
    
    w1, log_s1 = params[2], params[3]
    w2, log_s2 = params[4], params[5]
    w3, log_s3 = params[6], params[7]
    w4, log_s4 = params[8], params[9]
    
    # Centers
    # Using 4 centers in quadrants
    x1, y1 = -0.5, -0.5
    x2, y2 = 0.5, -0.5
    x3, y3 = -0.5, 0.5
    x4, y4 = 0.5, 0.5
    
    # Compute distances squared
    d1_sq = (x - x1)**2 + (y - y1)**2
    d2_sq = (x - x2)**2 + (y - y2)**2
    d3_sq = (x - x3)**2 + (y - y3)**2
    d4_sq = (x - x4)**2 + (y - y4)**2
    
    # Compute Gaussian kernels
    # s = exp(log_s), kernel = exp(-d_sq / s^2) = exp(-d_sq * exp(-2*log_s))
    # To avoid overflow in exp(-2*log_s) if log_s is very negative, 
    # we can compute the exponent directly.
    # exp_arg = -d_sq * exp(-2*log_s)
    # If log_s is large (wide), exp(-2*log_s) is small, exp_arg is close to 0.
    # If log_s is small (narrow), exp(-2*log_s) is large, exp_arg is very negative.
    
    # Safe calculation:
    # Let alpha = exp(-2 * log_s).
    # If log_s < -300, alpha overflows. 
    # However, BFGS starting from 1.0 (log_s=1, s=e) is safe.
    # We can clip log_s to prevent extreme values if necessary, but usually BFGS stays reasonable.
    # Let's use a soft approach: 
    # alpha = np.exp(-2.0 * log_s)
    # This can overflow. 
    # Alternative: kernel = np.exp(-d_sq / np.exp(2*log_s))
    # np.exp(2*log_s) can underflow to 0 if log_s is very negative.
    # If s=0, division by zero.
    # Better: kernel = np.exp(-d_sq * np.exp(-2*log_s))
    # If log_s is -100, exp(200) is huge. exp(-huge) is 0. This is fine.
    # If log_s is 100, exp(-200) is 0. exp(0) is 1. This is fine.
    
    a1 = np.exp(-2.0 * log_s1)
    a2 = np.exp(-2.0 * log_s2)
    a3 = np.exp(-2.0 * log_s3)
    a4 = np.exp(-2.0 * log_s4)
    
    k1 = np.exp(-d1_sq * a1)
    k2 = np.exp(-d2_sq * a2)
    k3 = np.exp(-d3_sq * a3)
    k4 = np.exp(-d4_sq * a4)
    
    return cx * x + cy * y + w1 * k1 + w2 * k2 + w3 * k3 + w4 * k4
