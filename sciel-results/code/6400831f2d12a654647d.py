"""
Strategy: Non-linear parametric model using a Sum of Gaussians (RBF-like).
Unlike the current quadratic polynomial which is linear in parameters and structurally limited to polynomial shapes,
this model uses parameters to define the centers (mu) and variances (sigma) of Gaussian basis functions.
This allows the model to capture non-linear features like peaks and shifts, which BFGS can optimize
even though the loss landscape is non-convex.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation: Sum of 3 Gaussian basis functions in 3D space.
    
    f(x,y,z) = sum_{i=0}^{2} w_i * exp(-((x-mu_xi)^2 + (y-mu_yi)^2 + (z-mu_zi)^2) / (2 * sigma_i^2))
    
    Parameters (10 total):
    - params[0:3] : Weights w_0, w_1, w_2
    - params[3:6] : Centers x-coordinates mu_x0, mu_x1, mu_x2
    - params[6:9] : Centers y-coordinates mu_y0, mu_y1, mu_y2
    - params[9]   : Common variance scale sigma^2 (shared to reduce complexity and improve conditioning)
    
    Note: We use a shared sigma to ensure the 10-parameter vector is fully utilized without 
    over-parameterizing the width, which can cause BFGS to get stuck in flat regions if widths 
    are independent and initially large/small.
    """
    # Extract parameters
    w0, w1, w2 = params[0], params[1], params[2]
    mx0, mx1, mx2 = params[3], params[4], params[5]
    my0, my1, my2 = params[6], params[7], params[8]
    sigma2 = params[9]
    
    # Ensure sigma2 is positive and not too small to avoid division by zero or overflow
    # If sigma2 is negative or near zero, we clamp it to a small positive value.
    # BFGS might explore negative values, so we need a safe mapping.
    # Using softplus-like behavior or just clamping. Clamping is simpler and robust for this context.
    safe_sigma2 = np.maximum(sigma2, 1e-6)
    
    # Calculate squared distances for each Gaussian
    # dist_i^2 = (x - mx_i)^2 + (y - my_i)^2 + (z - 1.0)^2 
    # We assume the z-center is fixed at 1.0 (middle of [0.5, 2.0]) to save parameters,
    # or we could use a fixed offset. Given we have 10 params:
    # 3 weights + 3 x-centers + 3 y-centers + 1 sigma = 10.
    # This leaves no parameter for z-centers. 
    # Alternative: 2 Gaussians?
    # 2 Gaussians: 2*w + 2*mx + 2*my + 2*mz + 1*sigma = 9 params. 1 left over.
    # Let's stick to 3 Gaussians but fix z-center to 1.0 (center of domain).
    # This is a structural assumption, but often valid for symmetric domains or if the function 
    # is not strongly z-dependent in a shifted way, or if the z-dependence is captured by 
    # the width. However, a more flexible approach for 10 params:
    
    # Let's try a different split to be more flexible:
    # 2 Gaussians in 3D.
    # Params: 
    # 0: w0
    # 1: w1
    # 2: mx0
    # 3: mx1
    # 4: my0
    # 5: my1
    # 6: mz0
    # 7: mz1
    # 8: sigma0
    # 9: sigma1
    
    # This uses all 10 params and allows different centers and widths for 2 components.
    # This is more robust than fixing z-center.
    
    w0, w1 = params[0], params[1]
    mx0, mx1 = params[2], params[3]
    my0, my1 = params[4], params[5]
    mz0, mz1 = params[6], params[7]
    s0, s1 = params[8], params[9]
    
    # Safe sigmas
    s0_safe = np.maximum(s0, 1e-6)
    s1_safe = np.maximum(s1, 1e-6)
    
    # Compute terms
    # Term 0
    dist0_sq = (x - mx0)**2 + (y - my0)**2 + (z - mz0)**2
    term0 = w0 * np.exp(-dist0_sq / (2.0 * s0_safe))
    
    # Term 1
    dist1_sq = (x - mx1)**2 + (y - my1)**2 + (z - mz1)**2
    term1 = w1 * np.exp(-dist1_sq / (2.0 * s1_safe))
    
    return term0 + term1
# EVOLVE-BLOCK-END
