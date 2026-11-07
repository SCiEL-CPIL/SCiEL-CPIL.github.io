"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 3D Gaussian RBFs + separable polynomial terms.
    10 params: [w1, w2, c1x, c1y, c1z, s1, c2x, c2y, c2z, s2]
    The polynomial part is fixed (not parameterized) to provide global structure,
    while the RBF part adapts locally.
    """
    p = params
    # Extract parameters
    w1 = p[0]
    w2 = p[1]
    c1x, c1y, c1z = p[2], p[3], p[4]
    s1 = p[5]
    c2x, c2y, c2z = p[6], p[7], p[8]
    s2 = p[9]
    
    # Ensure widths are positive and smooth
    s1_safe = np.sqrt(s1**2 + 1e-8)
    s2_safe = np.sqrt(s2**2 + 1e-8)
    
    # Compute squared distances
    d1_sq = (x - c1x)**2 + (y - c1y)**2 + (z - c1z)**2
    d2_sq = (x - c2x)**2 + (y - c2y)**2 + (z - c2z)**2
    
    # Compute Gaussian kernels with clipping for numerical stability
    exp1 = -d1_sq / (2.0 * s1_safe**2)
    exp2 = -d2_sq / (2.0 * s2_safe**2)
    
    # Clip to avoid underflow/overflow issues
    exp1 = np.clip(exp1, -500, 0)
    exp2 = np.clip(exp2, -500, 0)
    
    k1 = np.exp(exp1)
    k2 = np.exp(exp2)
    
    # Add a fixed polynomial component to capture global trends
    # This provides a baseline that the RBFs can correct locally
    poly = 0.5 * (x + y + z) + 0.1 * (x * y + y * z + x * z)
    
    return w1 * k1 + w2 * k2 + poly
# EVOLVE-BLOCK-END
