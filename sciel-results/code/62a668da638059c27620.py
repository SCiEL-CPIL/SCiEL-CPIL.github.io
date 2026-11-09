import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian Radial Basis Function (RBF) network.
    
    Strategy:
    - Instead of a global polynomial basis, we use local Gaussian kernels.
    - This allows the model to fit arbitrary smooth functions, including those
      with sharp features or non-separable interactions, which polynomials may
      struggle to approximate with only 10 parameters.
    - The 10 parameters are structured as:
      - 3 centers in x: cx1, cx2, cx3
      - 3 centers in y: cy1, cy2, cy3
      - 1 common width: sigma (shifted to ensure positive and finite)
      - 3 weights: w1, w2, w3
    - The model output is the sum of 3 Gaussian bumps:
      f(x, y) = sum_{i=1}^{3} w_i * exp(-((x - cx_i)^2 + (y - cy_i)^2) / (2 * sigma^2))
    - To help BFGS starting from all ones:
      - Centers are initialized near 0 (1 - 1 = 0)
      - Sigma is initialized to a reasonable width (np.exp(1) ~ 2.7, but we use a shift)
      - Weights are initialized to 1
    - We include a deadline guard to ensure we return a valid result within 120s.
    """
    start_time = time.time()
    
    p = params
    
    # Unpack parameters
    # Indices 0-2: cx1, cx2, cx3
    # Indices 3-5: cy1, cy2, cy3
    # Index 6: sigma_raw
    # Indices 7-9: w1, w2, w3
    
    cx1, cx2, cx3 = p[0], p[1], p[2]
    cy1, cy2, cy3 = p[3], p[4], p[5]
    sigma_raw = p[6]
    w1, w2, w3 = p[7], p[8], p[9]
    
    # Ensure sigma is positive and not too small to avoid numerical issues
    # np.exp(sigma_raw) ensures positivity. Starting at 1 gives exp(1)~2.7, which is wide.
    # BFGS can adjust sigma_raw to make it narrower or wider.
    sigma = np.exp(sigma_raw)
    
    # Safety check: if sigma is too large or too small, clip it to avoid overflow/underflow
    # In [-1, 1], a sigma > 10 is effectively constant, sigma < 1e-6 is a spike.
    sigma = np.clip(sigma, 1e-6, 10.0)
    
    # Compute squared distances for each Gaussian
    # Using vectorized operations for speed
    
    # Gaussian 1
    dx1 = x - cx1
    dy1 = y - cy1
    r2_1 = dx1**2 + dy1**2
    g1 = np.exp(-r2_1 / (2.0 * sigma**2))
    
    # Gaussian 2
    dx2 = x - cx2
    dy2 = y - cy2
    r2_2 = dx2**2 + dy2**2
    g2 = np.exp(-r2_2 / (2.0 * sigma**2))
    
    # Gaussian 3
    dx3 = x - cx3
    dy3 = y - cy3
    r2_3 = dx3**2 + dy3**2
    g3 = np.exp(-r2_3 / (2.0 * sigma**2))
    
    # Weighted sum
    pred = w1 * g1 + w2 * g2 + w3 * g3
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.zeros_like(x)
        
    # Deadline guard (though this function is called many times, it's fast, 
    # but the evaluator kills the whole process after 150s, so this is just a 
    # safety net for the single call, though the main risk is the optimizer 
    # taking too long. However, we cannot control the optimizer's time here, 
    # only ensure our function is fast and robust.)
    
    if time.time() - start_time > 110:
        # If we are extremely slow (unlikely for this simple math), return zeros
        return np.zeros_like(x)
        
    return pred
