import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize input to [-1, 1] for better conditioning
    t = x / 3.0
    
    # params[0]: Linear term coefficient (allows for overall slope/offset)
    # params[1..9]: 3 Gaussians with (amplitude, center, width)
    # Centers are in normalized t-space [-1, 1]
    # Widths are constrained to be positive and not too small to avoid numerical issues
    
    # 1. Linear/Polynomial part
    # We keep a simple linear term + constant to capture global trends
    # Note: We use t^2 as well to capture curvature if centers are symmetric
    # But to keep params distinct, let's just use a quadratic polynomial base
    # Actually, let's stick to the prompt's "polynomial terms" but adaptive.
    # However, adaptive centers for polynomials don't make sense.
    # So we keep a fixed low-order polynomial base for stability, 
    # and add adaptive Gaussians for the "unknown function" features.
    
    # Fixed base: p0=1, p1=t, p2=t^2
    P0 = np.ones_like(t)
    P1 = t
    P2 = t * t
    
    # Adaptive Gaussians
    # G_i(t) = A_i * exp(-((t - c_i) / w_i)^2)
    
    # Extract params
    A1 = params[0]
    c1 = params[1]
    w1 = np.abs(params[2]) + 0.1  # Ensure width >= 0.1
    
    A2 = params[3]
    c2 = params[4]
    w2 = np.abs(params[5]) + 0.1
    
    A3 = params[6]
    c3 = params[7]
    w3 = np.abs(params[8]) + 0.1
    
    # params[9] can be used for a global scale or another feature
    # Let's use it as a higher order polynomial coefficient to allow flexibility
    P3 = P2 * t
    A_poly3 = params[9]
    
    # Calculate Gaussians
    # Clip centers to avoid extreme values that might cause overflow in exp
    # t is in [-1, 1], centers likely in [-1, 1]
    # If center is far, exp(-large) -> 0, which is safe.
    
    G1 = np.exp(-((t - c1) / w1) ** 2)
    G2 = np.exp(-((t - c2) / w2) ** 2)
    G3 = np.exp(-((t - c3) / w3) ** 2)
    
    # Combine
    y = (A_poly3 * P3 + 
         A1 * G1 + 
         A2 * G2 + 
         A3 * G3)
         
    # We lost the constant and linear terms in the above combination if we only used P3 and Gaussians.
    # Wait, the previous program used 5 polys + 5 gaussians.
    # Here I have 1 poly (cubic) + 3 gaussians (9 params). Total 10 params.
    # This might be too rigid if the function has a strong linear component.
    # Let's refine: 
    # params[0]: constant
    # params[1]: linear
    # params[2]: quadratic
    # params[3..8]: 2 Gaussians (A, c, w) each -> 6 params
    # params[9]: maybe a 3rd Gaussian amplitude? Or keep 1 Gaussian and add more polys?
    
    # Let's try:
    # 3 Polynomials (1, t, t^2) -> params[0], params[1], params[2]
    # 2 Adaptive Gaussians (A, c, w) -> params[3..8]
    # 1 Adaptive Sigmoid or another Gaussian?
    # Let's use 2 Adaptive Gaussians and 1 Adaptive "Bump" (Gaussian) but maybe different width.
    # Actually, 3 Adaptive Gaussians is better for capturing multiple peaks/valleys.
    # But we need to ensure we can represent a constant/linear function.
    # A very wide Gaussian (w -> inf) acts like a constant.
    # So 3 Adaptive Gaussians can represent constants, linear (if centers shift), etc.
    # But BFGS might struggle to find "infinite" width.
    
    # Revised Plan:
    # params[0]: Constant term (fixed basis 1)
    # params[1]: Linear term (fixed basis t)
    # params[2]: Quadratic term (fixed basis t^2)
    # params[3..8]: Gaussian 1 (A, c, w)
    # params[9]: Amplitude of Gaussian 2, but share center/width? No, that's bad.
    
    # Let's go back to:
    # 1 Fixed Polynomial (t^2) to handle curvature? No, Gaussians handle curvature.
    # Let's use:
    # params[0]: Amplitude G1
    # params[1]: Center G1
    # params[2]: Width G1
    # params[3]: Amplitude G2
    # params[4]: Center G2
    # params[5]: Width G2
    # params[6]: Amplitude G3
    # params[7]: Center G3
    # params[8]: Width G3
    # params[9]: Global Linear Term (t)
    
    # This allows:
    # - 3 localized features (peaks/valleys)
    # - A global slope
    # - Constants are covered by wide Gaussians or if centers align.
    
    A1 = params[0]
    c1 = params[1]
    w1 = np.abs(params[2]) + 0.1
    
    A2 = params[3]
    c2 = params[4]
    w2 = np.abs(params[5]) + 0.1
    
    A3 = params[6]
    c3 = params[7]
    w3 = np.abs(params[8]) + 0.1
    
    Linear = params[9] * t
    
    G1 = np.exp(-((t - c1) / w1) ** 2)
    G2 = np.exp(-((t - c2) / w2) ** 2)
    G3 = np.exp(-((t - c3) / w3) ** 2)
    
    y = A1 * G1 + A2 * G2 + A3 * G3 + Linear
    
    return y
