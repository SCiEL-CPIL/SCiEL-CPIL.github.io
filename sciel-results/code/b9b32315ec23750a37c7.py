"""
Non-linear parametric model using Sum of Gaussians (RBF-like).
Parameters define centers and scales of 3 Gaussians in 3D space.
This allows capturing localized structures better than a global polynomial.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: sum of 3 Gaussian Radial Basis Functions.
    params layout:
    p0, p1, p2: center (cx0, cy0, cz0) for Gaussian 0
    p3:         scale (sigma0) for Gaussian 0
    p4, p5, p6: center (cx1, cy1, cz1) for Gaussian 1
    p7:         scale (sigma1) for Gaussian 1
    p8, p9:     center (cx2, cy2) for Gaussian 2 (z-center fixed to 1.25)
    p10:        scale (sigma2) for Gaussian 2
    Note: params length is 10 as per contract, so we use 3 Gaussians with slightly reduced params per Gaussian or fixed some centers.
    Let's use:
    G0: center (p0, p1, p2), scale p3
    G1: center (p4, p5, p6), scale p7
    G2: center (p8, p9, 1.25), scale 0.8 (fixed scale for last one to save params? No, we have 10 params exactly.)
    
    Revised layout for 10 params:
    p0, p1, p2, p3: G0 center (x,y,z), scale
    p4, p5, p6, p7: G1 center (x,y,z), scale
    p8, p9: G2 center (x,y), z=1.25 fixed, scale=0.8 fixed? 
    Actually, let's just use 2 full Gaussians (8 params) and 1 partial (2 params) or similar.
    Or better: 3 Gaussians, but fix some centers?
    Let's try:
    G0: center (p0, p1, p2), scale p3
    G1: center (p4, p5, p6), scale p7
    G2: center (p8, p9, 1.25), scale 0.8
    This uses 10 params.
    """
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    p4, p5, p6, p7 = params[4], params[5], params[6], params[7]
    p8, p9 = params[8], params[9]
    
    # Fixed parameters for G2 to save space
    cx2, cy2, cz2 = p8, p9, 1.25
    sigma2 = 0.8

    # Calculate distances squared
    dx0 = x - p0
    dy0 = y - p1
    dz0 = z - p2
    r0_sq = dx0*dx0 + dy0*dy0 + dz0*dz0
    
    dx1 = x - p4
    dy1 = y - p5
    dz1 = z - p6
    r1_sq = dx1*dx1 + dy1*dy1 + dz1*dz1
    
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    r2_sq = dx2*dx2 + dy2*dy2 + dz2*dz2

    # Compute Gaussians
    # Add small epsilon to sigma to avoid division by zero if optimizer tries sigma=0
    eps = 1e-6
    s0 = np.maximum(p3, eps)
    s1 = np.maximum(p7, eps)
    
    g0 = np.exp(-r0_sq / (2 * s0 * s0))
    g1 = np.exp(-r1_sq / (2 * s1 * s1))
    g2 = np.exp(-r2_sq / (2 * sigma2 * sigma2))
    
    # Sum with fixed weights (or we could make weights part of params, but we used all 10 for centers/scales)
    # If we fix weights to 1.0, the model is sum of 3 unit-amplitude Gaussians.
    # This might be restrictive. Let's see if we can free up a param.
    # Alternative: 2 Gaussians (8 params) + 2 linear terms?
    # Let's stick to 3 Gaussians with fixed weights first. If it fails, we can adjust.
    # Actually, amplitude is crucial. Let's use 1 Gaussian with 4 params (c, s) and 2 linear terms?
    # No, the prompt suggests RBF. Let's try 2 Gaussians (8 params) and 2 linear coefficients?
    # p0..p3: G0 (cx, cy, cz, s)
    # p4..p7: G1 (cx, cy, cz, s)
    # p8, p9: linear coeffs for x and y?
    
    # Let's restart the param layout for better expressiveness:
    # G0: center (p0, p1, p2), scale p3
    # G1: center (p4, p5, p6), scale p7
    # Linear: p8*x + p9*y
    
    g0 = np.exp(-(dx0*dx0 + dy0*dy0 + dz0*dz0) / (2 * s0 * s0))
    g1 = np.exp(-(dx1*dx1 + dy1*dy1 + dz1*dz1) / (2 * s1 * s1))
    
    return g0 + g1 + p8 * x + p9 * y

# EVOLVE-BLOCK-END
