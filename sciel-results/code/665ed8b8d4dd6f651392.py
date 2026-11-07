import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Learnable Gaussian RBF Network with BFGS Optimization
    
    Unlike the linear Chebyshev or fixed-knot spline approaches, this model treats
    the basis function parameters (centers and widths) as learnable variables.
    
    Key Differences:
    1. Non-Linear in Parameters: The centers (mu) and widths (sigma) are parameters
       optimized by BFGS. This allows the model to "find" the locations of features
       in the data (e.g., peaks, valleys, discontinuities) rather than relying on
       a fixed grid or global orthogonal basis.
    2. Local Adaptivity: Gaussian RBFs are local. A few well-placed RBFs can
       capture sharp local variations that a global polynomial of degree 9 might
       smooth out or oscillate poorly around (Runge's phenomenon).
    3. Structure: We use 3 RBFs to consume 9 parameters (3 centers, 3 widths) and 
       1 parameter for a global bias/offset. Total 10 parameters.
       - mu_1, mu_2, mu_3: Centers of the 3 Gaussians
       - sigma_1, sigma_2, sigma_3: Widths (standard deviations)
       - b: Bias term
       
       Wait, that's 7 parameters. We have 10. Let's use 4 RBFs?
       4 centers + 4 widths + 2 biases? Or 4 centers + 4 widths + 1 global scale + 1 bias?
       
       Let's stick to 3 RBFs but add a linear term?
       y = a1*exp(-(x-mu1)^2/(2*sig1^2)) + a2*exp(...) + a3*exp(...) + c0 + c1*x
       Params:
       0: mu1
       1: sig1
       2: a1
       3: mu2
       4: sig2
       5: a2
       6: mu3
       7: sig3
       8: a3
       9: bias
       
       This uses 9 parameters. We have 10.
       Let's add a linear slope?
       9: slope
       
       So: y = sum(ai * exp(-(x-mui)^2 / (2*sigi^2))) + bias + slope*x
       
       Why this might work better than Chebyshev:
       - Chebyshev is global. If the function is "spiky" or has distinct local features
         not aligned with polynomial trends, a global basis needs high degree to fit,
         leading to instability or poor generalization.
       - RBFs with learnable centers can snap to the "important" x-values.
       - BFGS is capable of navigating this non-convex landscape if the initial
         conditions are reasonable. Starting from all ones means centers=1, widths=1,
         amps=1. This is a reasonable start for [0.1, 6].
       
       Potential Issue: Local minima.
       Mitigation: The landscape of RBF fitting is often smooth enough for BFGS to
       converge to a good local minimum, especially with only 3-4 basis functions.
       The score improvement from "fixed" to "learnable" kernels was observed in
       other solutions (1.67 -> 6.085e+04), suggesting this structural change is
       high-value.
    """
    
    start_time = time.time()
    TIME_LIMIT = 145.0
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # Extract parameters
    # 0: mu1, 1: sig1, 2: a1
    # 3: mu2, 4: sig2, 5: a2
    # 6: mu3, 7: sig3, 8: a3
    # 9: bias
    # We will ignore a 10th parameter or use it for a linear term? 
    # Let's use param 9 as bias. We have 10 params total.
    # Actually, let's use 4 RBFs to be safer?
    # 4 centers + 4 widths + 2 amps? No, need amps for each.
    # 3 RBFs + Bias is 9 params. 
    # Let's add a linear term coefficient as param 9? 
    # y = RBFs + bias + p[9]*x
    
    mu1 = p[0]
    sig1 = p[1]
    a1 = p[2]
    mu2 = p[3]
    sig2 = p[4]
    a2 = p[5]
    mu3 = p[6]
    sig3 = p[7]
    a3 = p[8]
    bias = p[9]
    
    # Ensure widths are positive and not too small to avoid division by zero or overflow
    # We use a softplus-like transformation or just clip?
    # BFGS will try to optimize sig. If sig becomes negative, exp(-x^2/neg) explodes.
    # We must constrain sig > 0. 
    # Standard trick: sig = softplus(param) or exp(param) or abs(param)+eps.
    # Using exp() ensures positivity and smooth gradients.
    # However, exp(1) = 2.718. If the optimal width is 0.5, exp(-0.69) = 0.5.
    # This is fine.
    
    # But wait, if we use exp(sig), the gradient at sig=1 is exp(1)*d/dsig.
    # Let's use a safer transformation: sig = 0.1 + np.abs(sig_param)
    # This allows negative parameters to map to positive widths and is differentiable 
    # except at 0 (where subgradient is fine for BFGS usually, or we can use abs+eps).
    # Actually, np.abs is not differentiable at 0. 
    # Let's use: sig = 0.01 + 0.5 * (1 + np.tanh(sig_param))
    # This maps (-inf, inf) to (0.01, 1.01). This constrains width to [0.01, 1.01].
    # Is that enough? The domain is [0.1, 6]. A width of 1 covers about 2 units.
    # Maybe we want wider support?
    # Let's use: sig = 0.05 + 2.0 * sigmoid(sig_param)
    # Maps to (0.05, 2.05).
    
    # Alternative: Just clip the raw parameter to a safe range and hope BFGS stays there?
    # No, BFGS will step.
    
    # Let's use: sig = np.exp(0.5 * sig_param)
    # If sig_param=1, sig=1.64. If sig_param=0, sig=1. If sig_param=-2, sig=0.36.
    # This is smooth and positive.
    
    s1 = np.exp(0.5 * sig1)
    s2 = np.exp(0.5 * sig2)
    s3 = np.exp(0.5 * sig3)
    
    # Calculate RBFs
    # To avoid overflow in exp, we clip the exponent argument
    # arg = -(x-mu)^2 / (2*sig^2)
    
    # Vectorized calculation for efficiency
    # x is (N,), mu is scalar
    d1 = x - mu1
    d2 = x - mu2
    d3 = x - mu3
    
    # Prevent division by zero (s is always > 0.05 approx)
    # exp_arg = -d^2 / (2*s^2)
    
    # If s is very small, d^2/s^2 can be huge.
    # We clip the argument to [-500, 0] to prevent underflow/overflow issues in exp
    # exp(-500) is ~ 0. exp(0) is 1.
    
    arg1 = -(d1 * d1) / (2.0 * s1 * s1)
    arg2 = -(d2 * d2) / (2.0 * s2 * s2)
    arg3 = -(d3 * d3) / (2.0 * s3 * s3)
    
    # Clip to avoid numerical issues
    arg1 = np.clip(arg1, -500.0, 0.0)
    arg2 = np.clip(arg2, -500.0, 0.0)
    arg3 = np.clip(arg3, -500.0, 0.0)
    
    y = a1 * np.exp(arg1) + a2 * np.exp(arg2) + a3 * np.exp(arg3) + bias
    
    if not np.all(np.isfinite(y)):
        # Fallback to constant if numerical instability occurs
        return np.full_like(x, bias)
        
    return y
