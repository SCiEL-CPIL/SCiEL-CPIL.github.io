import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Use a physics-informed radial basis function (RBF) representation.
    
    The 10 parameters are interpreted as:
    - params[0:8]: Centers (cx, cy) for 4 Gaussian RBFs in the (x, v) plane.
      cx = params[0], params[2], params[4], params[6]
      cy = params[1], params[3], params[5], params[7]
    - params[8]: Common width (sigma) for all RBFs.
    - params[9]: Global scale/offset (or amplitude multiplier).
    
    The model is: y = params[9] * sum( exp(-((x-cx)^2 + (v-cy)^2) / (2*sigma^2)) )
    
    This allows the model to capture localized nonlinear features that global polynomials miss.
    The centers can move around the (x, v) space to fit the data.
    The width controls the locality.
    
    To ensure stability and differentiability for BFGS:
    - We clamp sigma to a minimum value to avoid division by zero or extreme gradients.
    - We scale the inputs and centers to be in a reasonable range (e.g., [-1, 1]) to keep
      the exponentials well-behaved.
    - The initial guess is all ones. So initially:
      centers: (1,1), (1,1), (1,1), (1,1)
      sigma = 1
      scale = 1
      The RBFs are centered at (1,1) with width 1. The data is roughly in [-0.6, 0.6] for x and [-0.3, 0.3] for v.
      So initially, the RBFs are outside the data range, leading to small values. BFGS will move the centers inward.
      
    To help BFGS, we can re-parameterize the centers to be bounded or use a softplus-like transformation?
    No, simple linear re-parameterization is usually fine for BFGS if the landscape is smooth.
    
    However, the landscape of RBF centers can be rugged if the centers overlap or if sigma is too small.
    A fixed sigma might be better? But we have 10 params.
    Let's use:
    - 4 centers (8 params)
    - 1 width (1 param)
    - 1 amplitude (1 param)
    
    If the width is too small, the function becomes very peaky, and gradients vanish or explode.
    If the width is too large, it becomes a global constant, losing locality.
    
    Let's try to constrain the width to be positive and reasonable.
    sigma = params[8]
    We can use a softplus or just clamp it. Clamping is non-differentiable at the boundary.
    Better: sigma = 0.1 + 0.9 * sigmoid(params[8])? Or just sigma = abs(params[8]) + 0.1?
    Abs is not differentiable at 0.
    Let's use: sigma = 0.1 + 0.5 * (1 + np.tanh(params[8]))  # Range [0.1, 0.6]
    This keeps sigma in a reasonable range for the input scale.
    
    Similarly, centers should be within the data range.
    cx_i = 0.6 * np.tanh(params[2*i])
    cy_i = 0.3 * np.tanh(params[2*i+1])
    
    Amplitude: A = params[9]
    
    Model:
    y = A * sum_{i=1 to 4} exp(-((x - cx_i)^2 + (v - cy_i)^2) / (2 * sigma^2))
    
    This is smooth and differentiable everywhere.
    The parameters are bounded in their effect, which helps BFGS.
    The initial guess (all ones) will map to:
    cx_i = 0.6 * tanh(1) ≈ 0.42
    cy_i = 0.3 * tanh(1) ≈ 0.21
    sigma = 0.1 + 0.5 * (1 + tanh(1)) ≈ 0.1 + 0.5 * 1.76 ≈ 0.98
    A = 1
    
    This places the RBFs near the center of the data, with a reasonable width.
    """
    # Extract parameters
    # Centers
    cx1 = 0.6 * np.tanh(params[0])
    cy1 = 0.3 * np.tanh(params[1])
    cx2 = 0.6 * np.tanh(params[2])
    cy2 = 0.3 * np.tanh(params[3])
    cx3 = 0.6 * np.tanh(params[4])
    cy3 = 0.3 * np.tanh(params[5])
    cx4 = 0.6 * np.tanh(params[6])
    cy4 = 0.3 * np.tanh(params[7])
    
    # Width
    sigma = 0.1 + 0.5 * (1.0 + np.tanh(params[8]))
    
    # Amplitude
    A = params[9]
    
    # Compute RBFs
    # Vectorized computation
    # x and v are 1D arrays
    r2_1 = (x - cx1)**2 + (v - cy1)**2
    r2_2 = (x - cx2)**2 + (v - cy2)**2
    r2_3 = (x - cx3)**2 + (v - cy3)**2
    r2_4 = (x - cx4)**2 + (v - cy4)**2
    
    denom = 2.0 * sigma**2
    
    phi1 = np.exp(-r2_1 / denom)
    phi2 = np.exp(-r2_2 / denom)
    phi3 = np.exp(-r2_3 / denom)
    phi4 = np.exp(-r2_4 / denom)
    
    return A * (phi1 + phi2 + phi3 + phi4)
