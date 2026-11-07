"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Gaussian RBF basis with learnable centers and widths.
    
    The function is a sum of 3 Gaussian kernels in the 3D input space (t, x, v).
    Parameters (length 10):
    - p0, p1, p2: Weights for the 3 kernels
    - p3, p4, p5: Centers (t_c, x_c, v_c) for Kernel 1
    - p6, p7, p8: Centers (t_c, x_c, v_c) for Kernel 2
    - p9: Width (sigma) shared by all kernels (or could be separate, but 10 params limits us)
    
    To maximize flexibility within 10 params, we use:
    - 3 weights
    - 3 centers for a single dominant kernel? No, that's too rigid.
    
    Let's use 2 kernels to allow more center variety.
    Params:
    - w1, w2, w3 (3 weights)
    - c1_t, c1_x, c1_v (3 centers for kernel 1)
    - c2_t, c2_x, c2_v (3 centers for kernel 2)
    - sigma (1 width)
    Total: 3 + 3 + 3 + 1 = 10 params.
    
    This allows the optimizer to place two "blobs" of influence in the state space.
    """
    # Deadline guard
    start_time = time.time()
    
    # Unpack parameters
    w1, w2, w3 = params[0], params[1], params[2]
    c1_t, c1_x, c1_v = params[3], params[4], params[5]
    c2_t, c2_x, c2_v = params[6], params[7], params[8]
    sigma = params[9]
    
    # Prevent division by zero or extreme values in exp
    # If sigma is very small, the kernel becomes a spike. If very large, it becomes flat.
    # BFGS starting from all ones should be okay, but let's ensure numerical stability.
    sigma_safe = np.clip(sigma, 1e-3, 1e3)
    
    # Calculate squared distances
    # Kernel 1
    dist1_sq = (t - c1_t)**2 + (x - c1_x)**2 + (v - c1_v)**2
    # Kernel 2
    dist2_sq = (t - c2_t)**2 + (x - c2_x)**2 + (v - c2_v)**2
    
    # Gaussian kernels
    k1 = np.exp(-dist1_sq / (2.0 * sigma_safe**2))
    k2 = np.exp(-dist2_sq / (2.0 * sigma_safe**2))
    
    # Combine
    # We have 3 weights but only 2 kernels. 
    # Let's adjust the strategy: 
    # Use 3 kernels but share centers? No.
    # Let's use 3 kernels with fixed centers and learnable weights/widths?
    # Or 2 kernels with learnable centers and 1 global bias/width?
    
    # Actually, 10 params is tight for 3 full kernels.
    # Let's stick to 2 full kernels (6 center params) + 2 weights + 1 sigma + 1 bias?
    # Total: 6 + 2 + 1 + 1 = 10.
    
    # Re-mapping:
    # p0: bias
    # p1, p2: weights for kernel 1, 2
    # p3, p4, p5: center 1 (t, x, v)
    # p6, p7, p8: center 2 (t, x, v)
    # p9: sigma
    
    bias = params[0]
    w1 = params[1]
    w2 = params[2]
    c1_t, c1_x, c1_v = params[3], params[4], params[5]
    c2_t, c2_x, c2_v = params[6], params[7], params[8]
    sigma = params[9]
    
    sigma_safe = np.clip(sigma, 1e-3, 1e3)
    
    dist1_sq = (t - c1_t)**2 + (x - c1_x)**2 + (v - c1_v)**2
    dist2_sq = (t - c2_t)**2 + (x - c2_x)**2 + (v - c2_v)**2
    
    k1 = np.exp(-dist1_sq / (2.0 * sigma_safe**2))
    k2 = np.exp(-dist2_sq / (2.0 * sigma_safe**2))
    
    result = bias + w1 * k1 + w2 * k2
    
    # Safety check for time
    if time.time() - start_time > 115:
         # Return a simple fallback if we somehow get stuck, though this is vectorized and fast
         return result
         
    return result
# EVOLVE-BLOCK-END
