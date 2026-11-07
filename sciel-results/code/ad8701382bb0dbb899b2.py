import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Multi-Resolution Gaussian RBF basis.
    
    Strategy:
    - Uses the 10 available parameters as the centers of 10 Gaussian kernels.
    - The kernels have fixed, hierarchical widths (sigma) to allow both broad
      and local approximations.
    - This is a local adaptive approach, structurally different from the global
      Chebyshev basis. It can fit sharp features or local variations better.
    - The weights are fixed to 1.0 to simplify the optimization landscape for BFGS,
      relying on the centers to adjust the shape. However, usually RBFs have
      learnable weights. With 10 params, we can't have 10 centers + 10 weights.
      So we must fix either centers or weights.
      
      Wait, if we fix weights to 1, the function is sum of Gaussians.
      If we fix centers, the function is linear in weights.
      
      Let's look at the inspirations.
      Inspiration 2 (score 4089536): "10 parameters act as the centers of Gaussian..."
      This implies weights were fixed or derived.
      If weights are fixed to 1, the model is f(x) = sum exp(-((x-c_i)^2)/(2*sigma_i^2)).
      This is a sum of bumps. It can approximate many functions if the target is positive and smooth.
      But if the target has negative values or oscillations, sum of positive bumps might struggle.
      
      Alternative: Use a smaller number of centers with weights?
      E.g., 3 centers, 3 widths, 3 weights, 1 bias? That's 10 params.
      But widths and centers are non-linear.
      
      Let's try: 4 Gaussian kernels.
      Params: c1, c2, c3, c4 (centers)
              w1, w2, w3, w4 (weights)
              sigma (shared width)
              bias
              That's 4+4+1+1 = 10 params.
              
      This allows positive and negative contributions (via weights) and a global offset (bias).
      The shared width sigma ensures all kernels have the same scale, which might be limiting,
      but it's a reasonable compromise. Or we can have two scales.
      
      Let's try:
      2 scales: sigma1, sigma2.
      3 centers for sigma1, 3 centers for sigma2?
      Params:
      c1, c2, c3 (centers for narrow)
      w1, w2, w3 (weights for narrow)
      c4, c5 (centers for wide)
      w4, w5 (weights for wide)
      sigma1, sigma2
      Total: 3+3+2+2+2 = 12. Too many.
      
      Let's stick to the Inspiration 2 idea but refine it.
      "10 parameters act as the centers of Gaussian..."
      If I use 10 centers, I have no weights.
      f(x) = sum_{i=1}^{10} exp(-((x-c_i)^2)/(2*sigma_i^2))
      If sigma_i are fixed and small, it's a sum of bumps.
      If the target function is, say, sin(x), a sum of positive bumps can approximate it
      if we allow negative "bumps"? No, exp is always positive.
      So sum of positive Gaussians can only approximate non-negative functions?
      Or functions that can be decomposed into positive lobes?
      sin(x) is negative in some parts. A sum of positive Gaussians cannot go below 0.
      So this is a very strong constraint.
      
      Better approach:
      Use 5 Gaussian kernels.
      Params:
      c1..c5 (centers)
      w1..w5 (weights)
      Total 10 params.
      Fix sigma to a reasonable value, e.g., 0.5.
      This allows positive and negative values.
      f(x) = sum w_i * exp(-((x-c_i)^2)/(2*0.5^2))
      
      This is a standard RBF model.
      The non-linearity comes from the centers.
      BFGS should handle this, though it's non-convex.
      Starting from all ones:
      c_i = 1.0 for all i.
      w_i = 1.0 for all i.
      This means 5 identical Gaussians at x=1.0 with weight 1.
      f(x) = 5 * exp(-((x-1)^2)/(2*0.25))
      This is a single big bump at x=1.
      
      Is this better than Chebyshev?
      Chebyshev failed (score 1e11).
      RBF with fixed centers (linear in weights) would have worked better than Chebyshev if the function is smooth, but Chebyshev is also a global basis.
      Actually, Chebyshev with 10 terms is a degree 9 polynomial.
      A degree 9 polynomial is quite flexible.
      Why did it fail?
      Maybe the function is not smooth? Or has high frequency?
      Or maybe the initialization (all ones) led to a bad local minimum?
      BFGS starting from all ones for Chebyshev coefficients:
      f(x) = sum T_i(x/2).
      T_0=1, T_1=x/2, T_2=2(x/2)^2-1 = x^2/2 - 1.
      Sum of T_i is a specific polynomial.
      
      Let's try the 5-kernel RBF with fixed sigma.
      To improve robustness, we can use two different sigmas.
      Params:
      c1, c2, c3 (centers for narrow, sigma=0.2)
      w1, w2, w3 (weights for narrow)
      c4, c5 (centers for wide, sigma=1.0)
      w4, w5 (weights for wide)
      Total: 3+3+2+2 = 10.
      
      This gives local detail (narrow) and global trend (wide).
      
      Let's implement this.
    """
    start_time = time.time()
    
    x = np.asarray(x).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([])
        
    # Ensure params is flat
    p = np.asarray(params).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant', constant_values=1.0)
    else:
        p = p[:10]
        
    # Extract parameters
    # Narrow kernels (3)
    c1, c2, c3 = p[0], p[1], p[2]
    w1, w2, w3 = p[3], p[4], p[5]
    
    # Wide kernels (2)
    c4, c5 = p[6], p[7]
    w4, w5 = p[8], p[9]
    
    # Fixed sigmas
    sigma_narrow = 0.2
    sigma_wide = 1.0
    
    # Compute Gaussian terms
    # To avoid overflow/underflow issues, we can clip the exponent
    # exp(-x) is safe for x > -700 roughly.
    # (x-c)^2 / (2*sigma^2) can be large.
    
    def gauss(x, c, sigma):
        d = x - c
        # Clip d to prevent overflow in square? No, d is finite.
        # The exponent is -d^2 / (2*sigma^2).
        # If d is large, exponent is very negative, exp is 0.
        # This is fine.
        return np.exp(-d**2 / (2.0 * sigma**2))
        
    try:
        y_narrow = w1 * gauss(x, c1, sigma_narrow) + \
                   w2 * gauss(x, c2, sigma_narrow) + \
                   w3 * gauss(x, c3, sigma_narrow)
                   
        y_wide = w4 * gauss(x, c4, sigma_wide) + \
                 w5 * gauss(x, c5, sigma_wide)
                 
        result = y_narrow + y_wide
        
    except Exception:
        return np.zeros_like(x)
        
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
