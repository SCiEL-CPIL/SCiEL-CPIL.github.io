import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a Gaussian Process-inspired kernel basis.
    
    Model:
    a = p0*X + p1*V 
      + p2 * exp(-0.5 * ((X-p3)^2 + (V-p4)^2) / (p5^2 + eps))
      + p6 * exp(-0.5 * ((X-p7)^2 + (V-p8)^2) / (p9^2 + eps))
      
    Inputs are scaled to O(1) range.
    p0, p1: Linear coefficients for scaled x and v.
    p2-p5: Amplitude, center_x, center_v, scale for first Gaussian bump.
    p6-p9: Amplitude, center_x, center_v, scale for second Gaussian bump.
    """
    # Deadline guard
    start_time = time.time()
    
    # Scaling constants based on training ranges
    # x in [-0.2056, 0.1749] -> mean ~ -0.015, std ~ 0.19
    # v in [-0.3391, 0.3624] -> mean ~ 0.01, std ~ 0.35
    # t in [30, 50] -> not used directly in Gaussian centers to save params, 
    # but we could use it if we had more params. For now, assume a(x,v).
    
    mu_x = -0.015
    sigma_x = 0.19
    mu_v = 0.01
    sigma_v = 0.35
    
    # Scale inputs
    X = (x - mu_x) / sigma_x
    V = (v - mu_v) / sigma_v
    
    # Extract parameters
    p0, p1 = params[0], params[1]
    A1, Cx1, Cv1, S1 = params[2], params[3], params[4], params[5]
    A2, Cx2, Cv2, S2 = params[6], params[7], params[8], params[9]
    
    # Safety margin for time (150s limit, keep under 145s)
    if time.time() - start_time > 140:
        # Return a simple linear prediction if running out of time
        return p0 * X + p1 * V

    eps = 1e-6
    
    # Compute Gaussian terms
    # Use S^2 to ensure positive variance, add eps to avoid division by zero
    S1_sq = S1 * S1 + eps
    S2_sq = S2 * S2 + eps
    
    diff1_x = X - Cx1
    diff1_v = V - Cv1
    r1_sq = diff1_x * diff1_x + diff1_v * diff1_v
    G1 = np.exp(-0.5 * r1_sq / S1_sq)
    
    diff2_x = X - Cx2
    diff2_v = V - Cv2
    r2_sq = diff2_x * diff2_x + diff2_v * diff2_v
    G2 = np.exp(-0.5 * r2_sq / S2_sq)
    
    # Combine
    pred = p0 * X + p1 * V + A1 * G1 + A2 * G2
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        # Fallback to linear if numerical issues occur
        pred = p0 * X + p1 * V
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred

# EVOLVE-BLOCK-END
