"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian Radial Basis Function (RBF) approximation.
    
    Strategy:
    Instead of using a fixed linear basis (like polynomials or fixed Fourier terms),
    we use a nonlinear basis where the parameters themselves define the geometry 
    of the basis functions (centers and widths). This allows the optimizer to 
    place "bumps" where the function has high curvature or specific features,
    which is often more effective for arbitrary smooth functions than fixed bases.
    
    The 10 parameters are mapped as follows:
    - p[0], p[1]: Center (cx1, cy1) of RBF 1
    - p[2], p[3]: Center (cx2, cy2) of RBF 2
    - p[4], p[5]: Center (cx3, cy3) of RBF 3
    - p[6], p[7]: Center (cx4, cy4) of RBF 4
    - p[8], p[9]: Centers (cx5, cy5) of RBF 5 is fixed at (0,0) to save params? 
      Actually, let's use 4 RBFs with centers and 2 global parameters?
      
    Let's refine the parameter mapping to ensure we have enough flexibility 
    while keeping the count at 10.
    
    Proposed Mapping:
    - 4 RBFs.
    - Each RBF has a center (cx, cy) and a width (sigma). That's 3 params per RBF -> 12 params. Too many.
    
    Alternative Mapping:
    - 5 RBFs.
    - Fixed centers on a grid? No, that defeats the purpose of adaptivity.
    
    Let's try:
    - 4 RBFs.
    - Centers: cx_i, cy_i for i=1..4. (8 params)
    - Widths: Shared width? Or derived?
    
    If we use 8 params for centers, we have 2 left.
    Let p[8] be a global width parameter (inverse sigma squared).
    Let p[9] be a global bias/offset.
    
    This allows the centers to move anywhere in [-1, 1]^2 (via sigmoid/tanh scaling)
    and the width to adapt globally.
    
    Centers: cx_i = tanh(p[2*i]) * 1.0, cy_i = tanh(p[2*i+1]) * 1.0
    Width: sigma_inv = softplus(p[8]) + 1e-6
    Bias: p[9]
    
    Prediction: sum(exp(-sigma_inv * ||x_i - c_i||^2)) + bias
    
    This is a nonlinear function of params, which is fine for BFGS.
    The landscape is non-convex, but BFGS might find local minima that fit well.
    To help convergence, we can initialize params such that centers are spread out.
    However, the evaluator starts from all ones.
    tanh(1) ~ 0.76. So centers will start near (0.76, 0.76), (0.76, 0.76)... 
    This is a bad initialization (all centers clustered).
    
    We need to break symmetry. 
    We can use different scaling factors or offsets for the tanh calls to spread initial centers.
    cx1 = tanh(p[0] * 1.0 + 1.0) # Start higher?
    Actually, simpler: just use the raw params scaled, but ensure they start different.
    Since we can't change the start, we can use different coefficients in the tanh argument.
    
    cx1 = tanh(p[0] + 1.0)
    cy1 = tanh(p[1] + 0.5)
    cx2 = tanh(p[2] - 1.0)
    cy2 = tanh(p[3] - 0.5)
    cx3 = tanh(p[4] + 0.5)
    cy3 = tanh(p[5] - 1.0)
    cx4 = tanh(p[6] - 0.5)
    cy4 = tanh(p[7] + 1.0)
    
    This spreads the initial centers.
    
    Width: p[8] controls the decay. exp(-exp(p[8]) * dist^2).
    If p[8]=1, exp(1)~2.7. dist^2 max is 4 (corners). exp(-10.8) is small.
    This creates localized bumps.
    
    Bias: p[9]
    
    Let's implement this.
    """
    p = params
    
    # Extract and transform parameters to ensure valid centers and positive width
    # Using tanh to keep centers roughly within [-1, 1] domain
    # Adding constants to break symmetry from the all-ones start
    
    cx1 = np.tanh(p[0] + 1.0)
    cy1 = np.tanh(p[1] + 0.5)
    
    cx2 = np.tanh(p[2] - 1.0)
    cy2 = np.tanh(p[3] - 0.5)
    
    cx3 = np.tanh(p[4] + 0.5)
    cy3 = np.tanh(p[5] - 1.0)
    
    cx4 = np.tanh(p[6] - 0.5)
    cy4 = np.tanh(p[7] + 1.0)
    
    # Width parameter: use softplus to ensure positive inverse variance
    # inv_sigma2 = exp(p[8]) ensures it's positive and differentiable
    # If p[8] is 1, inv_sigma2 = 2.718. 
    # Distance squared in [-1,1]^2 is [0, 4].
    # Max exponent = -2.718 * 4 = -10.8. This is quite narrow.
    # Maybe scale it down? inv_sigma2 = 0.5 * exp(p[8])?
    # Or just exp(p[8]) / 2.0
    inv_sigma2 = 0.5 * np.exp(p[8])
    
    bias = p[9]
    
    # Compute squared distances
    # x and y are arrays of shape (N,)
    dx1 = x - cx1
    dy1 = y - cy1
    d1_sq = dx1**2 + dy1**2
    
    dx2 = x - cx2
    dy2 = y - cy2
    d2_sq = dx2**2 + dy2**2
    
    dx3 = x - cx3
    dy3 = y - cy3
    d3_sq = dx3**2 + dy3**2
    
    dx4 = x - cx4
    dy4 = y - cy4
    d4_sq = dx4**2 + dy4**2
    
    # Compute Gaussian RBFs
    # Clip the exponent to avoid underflow/overflow issues, though exp(-large) is just 0
    # exp(-large) is safe in numpy (returns 0). exp(+large) would be bad, but we have negative sign.
    rbf1 = np.exp(-inv_sigma2 * d1_sq)
    rbf2 = np.exp(-inv_sigma2 * d2_sq)
    rbf3 = np.exp(-inv_sigma2 * d3_sq)
    rbf4 = np.exp(-inv_sigma2 * d4_sq)
    
    # Combine with equal weights (since we don't have enough params for individual weights)
    # This is a limitation, but 4 centers + 1 width + 1 bias = 10 params.
    # If we used individual weights, we'd need 4*3 + 4 = 16 params.
    # So we assume equal amplitude for all bumps.
    pred = (rbf1 + rbf2 + rbf3 + rbf4) + bias
    
    return pred
# EVOLVE-BLOCK-END
