"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 2 kernels.
    params layout (10 elements):
      params[0] = weight for kernel 1
      params[1] = weight for kernel 2
      params[2] = global offset
      params[3], params[4], params[5] = center of kernel 1 (cx1, cy1, cz1)
      params[6], params[7], params[8] = center of kernel 2 (cx2, cy2, cz2)
      params[9] = shared width (inverse bandwidth, positive)
    
    To ensure width is positive, we use exp(params[9]) internally.
    Centers are clamped to a reasonable range to avoid extreme values during optimization.
    """
    p = params
    
    # Extract parameters
    w1 = p[0]
    w2 = p[1]
    offset = p[2]
    
    cx1 = np.clip(p[3], 0.5, 2.0)
    cy1 = np.clip(p[4], 0.5, 2.0)
    cz1 = np.clip(p[5], 0.5, 2.0)
    
    cx2 = np.clip(p[6], 0.5, 2.0)
    cy2 = np.clip(p[7], 0.5, 2.0)
    cz2 = np.clip(p[8], 0.5, 2.0)
    
    # Width: use exp to ensure positivity, but clip to avoid overflow/underflow
    # exp(p[9]) where p[9] is around 1 initially -> ~2.7
    # We want the width to be such that the kernel is not too narrow or too wide.
    # Let sigma_inv = exp(p[9]). The exponent in Gaussian is -0.5 * sigma_inv^2 * dist^2
    # If sigma_inv is large, kernel is narrow. If small, wide.
    # Let's define the kernel as exp(-0.5 * (dist^2) / (sigma^2)) = exp(-0.5 * sigma_inv^2 * dist^2)
    # So we need sigma_inv.
    # To keep gradients stable, we can clip p[9] before exp.
    sigma_inv = np.exp(np.clip(p[9], -5.0, 5.0))
    
    # Compute squared distances for kernel 1
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    dist1_sq = dx1**2 + dy1**2 + dz1**2
    
    # Compute squared distances for kernel 2
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    dist2_sq = dx2**2 + dy2**2 + dz2**2
    
    # Gaussian kernels
    k1 = np.exp(-0.5 * sigma_inv**2 * dist1_sq)
    k2 = np.exp(-0.5 * sigma_inv**2 * dist2_sq)
    
    # Linear combination
    result = w1 * k1 + w2 * k2 + offset
    
    # Ensure finite values
    # If any values are non-finite, replace with 0 (or a safe value)
    # But np.exp of large negative numbers underflows to 0, which is fine.
    # Overflow only happens if sigma_inv is huge and dist is 0, but exp(0)=1.
    # So result should be finite as long as inputs are finite.
    
    return result
# EVOLVE-BLOCK-END
