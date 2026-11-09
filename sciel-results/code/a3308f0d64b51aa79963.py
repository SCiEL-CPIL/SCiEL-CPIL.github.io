"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a Radial Basis Function (RBF) network
    with fixed centers spanning the input domain.
    
    Structure:
    - 10 Gaussian kernels with fixed centers evenly spaced in [0.1, 6].
    - The width (sigma) of the Gaussians is determined by params[0].
      We use a softplus-like mapping or simple abs to ensure sigma > 0.
      Specifically, sigma = 0.1 + abs(params[0]) * 2.0. 
      This allows the optimizer to adjust the localization.
    - The remaining 9 parameters (params[1:10]) are the weights for the 10 kernels?
      Wait, we have 10 params total.
      If we use 10 kernels, we need 10 weights. But we also need a width.
      Let's use 9 kernels to have 1 param for width and 9 for weights? 
      Or 10 kernels and 1 param for width, but that's 11 params.
      
      Let's stick to 10 params.
      Option A: 10 kernels, width fixed. 
      Option B: 9 kernels, 1 width param, 9 weight params. Total 10.
      Option C: 10 kernels, width derived from one weight? No.
      
      Let's try Option B: 9 Gaussian kernels + 1 width parameter.
      Centers: 9 points in [0.1, 6].
      Width: sigma = 0.5 + abs(params[0]) * 2.0  (ensures positive, scales reasonably)
      Weights: params[1:10]
      
      Alternatively, maybe a mixture of RBF and a global polynomial term?
      The inspirations suggest hybrid bases work better.
      
      Let's try:
      - 1 param for global linear trend (to handle large-scale behavior)
      - 9 params for 9 Gaussian kernels (localized features)
      
      This gives 10 params total.
      y = params[0] * x + sum_{i=0}^{8} params[i+1] * exp(-((x - c_i)^2) / (2 * sigma^2))
      
      What is sigma?
      If sigma is fixed, we lose flexibility in localization.
      If sigma is learnable, we need another param.
      
      Let's look at inspiration 2 and 3. They used mixed bases.
      Inspiration 2: Poly + Trig + Exp. Score ~1.3e6.
      Inspiration 3: Cheb + Exp + Trig. Score ~2.2e4.
      Current: Cheb. Score ~1.2e9.
      
      The RBF directive suggests switching to RBF.
      Let's implement a pure RBF with a learnable width using a softplus of one param,
      and the rest as weights. But we only have 10 params.
      
      If we use 9 kernels, we have 9 weights. That leaves 1 param for width.
      Centers: np.linspace(0.1, 6, 9)
      sigma = 0.1 + np.exp(params[0])  # ensures sigma > 0.1
      y = sum(params[i+1] * exp(-0.5 * ((x - c_i)/sigma)^2) for i in 0..8)
      
      This is a valid structural change. It is localized.
      However, if the function is smooth and global (like a polynomial), RBFs might struggle
      if sigma is too small. If sigma is too large, it becomes like a polynomial.
      BFGS should be able to find a large sigma if needed.
      
      Let's add a constant term? No, that's another param.
      Maybe the first kernel acts as a constant if centered appropriately?
      
      Let's try a hybrid:
      Param 0: Width sigma (via exp)
      Params 1-9: Weights for 9 kernels.
      
      Actually, let's use 10 kernels and fix the width?
      If the width is fixed, we can't adapt to different scales.
      
      Let's try:
      Param 0: Coefficient for a global linear term (x)
      Params 1-9: Weights for 9 Gaussian kernels with a FIXED width.
      What width? The domain is [0.1, 6], range 5.9.
      9 centers. Spacing ~0.66.
      A width of ~0.5 to 1.0 seems reasonable.
      Let's fix sigma = 0.8.
      
      y = params[0] * x + sum_{i=0}^{8} params[i+1] * exp(-0.5 * ((x - c_i)/0.8)^2)
      
      This is simple and robust. The linear term handles the global trend,
      the RBFs handle local deviations.
      
      Let's refine the centers.
      c_i = 0.1 + i * (6 - 0.1) / 8 for i=0..8
      c = [0.1, 0.8375, 1.575, 2.3125, 3.05, 3.7875, 4.525, 5.2625, 6.0]
      
      Let's implement this.
    """
    x = np.asarray(x, dtype=float)
    
    # Fixed centers for 9 Gaussian kernels
    # Spaced evenly across [0.1, 6]
    centers = np.linspace(0.1, 6.0, 9)
    
    # Fixed width (sigma)
    sigma = 0.8
    sigma_sq = 2.0 * sigma * sigma
    
    # Initialize output
    y = np.zeros_like(x, dtype=float)
    
    # Global linear term
    # This helps BFGS converge faster for functions with a strong linear component
    y += params[0] * x
    
    # RBF terms
    # Vectorized computation for efficiency
    # x shape: (N,), centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-0.5 * (diff/sigma)^2)
    # We compute exp(-diff^2 / (2*sigma^2))
    # To avoid overflow/underflow issues, we can clip the exponent or rely on NumPy's stability
    # Given x in [0.1, 6] and centers in same range, max diff is ~6.
    # (6/0.8)^2 / 2 ~ 28. exp(-28) is very small but representable.
    # Min diff is 0. exp(0) = 1.
    
    kernels = np.exp(-diff * diff / sigma_sq)
    
    # weights are params[1:10]
    weights = params[1:10]
    
    # y += kernels @ weights
    y += kernels @ weights
    
    return y
# EVOLVE-BLOCK-END
