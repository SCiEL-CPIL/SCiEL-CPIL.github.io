import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a sum of two Gaussian bumps 
    plus a linear trend.
    
    Model:
    f(x, y) = b + wx*x + wy*y + A1*exp(-((x-cx1)^2 + (y-cy1)^2) / (2*sigma^2)) 
             + A2*exp(-((x-cx2)^2 + (y-cy2)^2) / (2*sigma^2))
             
    Parameters are transformed to ensure valid ranges:
    - Centers (cx, cy) are mapped to [-1, 1] using tanh.
    - Width (sigma) is mapped to (0, inf) using exp.
    - Amplitudes and linear terms are unconstrained.
    """
    # Unpack and transform parameters
    b = params[0]
    wx = params[1]
    wy = params[2]
    
    A1 = params[3]
    cx1 = np.tanh(params[4])
    cy1 = np.tanh(params[5])
    
    A2 = params[6]
    cx2 = np.tanh(params[7])
    cy2 = np.tanh(params[8])
    
    # Ensure sigma is positive and not too small/large to cause numerical issues
    # exp(p) for p in [-10, 10] is safe.
    # If params[9] is very negative, exp is 0. Division by 0?
    # We add a small epsilon or clip.
    # Let's use sigma = exp(params[9]) + 1e-6
    sigma = np.exp(params[9]) + 1e-6
    
    # Calculate squared distances
    dx1 = x - cx1
    dy1 = y - cy1
    dist1_sq = dx1 * dx1 + dy1 * dy1
    
    dx2 = x - cx2
    dy2 = y - cy2
    dist2_sq = dx2 * dx2 + dy2 * dy2
    
    # Calculate Gaussian terms
    # Use np.exp with clipping to avoid underflow warnings? 
    # np.exp handles underflow to 0 silently usually.
    g1 = np.exp(-dist1_sq / (2.0 * sigma * sigma))
    g2 = np.exp(-dist2_sq / (2.0 * sigma * sigma))
    
    pred = b + wx * x + wy * y + A1 * g1 + A2 * g2
    
    return pred
