"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Gaussian RBF basis.
    
    The function is a linear combination of 10 Gaussian kernels centered at
    fixed, representative points in the (t, x, v) space. This approach:
    1. Avoids the ill-conditioning of high-order polynomial terms (e.g., t^5, x^3*v).
    2. Provides local flexibility to capture nonlinearities and potential discontinuities.
    3. Is robust to BFGS optimization starting from all-ones because the basis functions
       are normalized and relatively orthogonal in their local supports.
    
    Inputs:
    t: time [30, 50]
    x: position [-0.2056, 0.1749]
    v: velocity [-0.3391, 0.3624]
    """
    
    # Fixed centers for the RBF basis functions.
    # We spread them across the known input ranges to ensure coverage.
    # t range: 30 to 50. Centers: 30, 40, 50.
    # x range: -0.2 to 0.2. Centers: -0.2, 0, 0.2.
    # v range: -0.34 to 0.36. Centers: -0.34, 0, 0.36.
    
    # To use exactly 10 params, we select 10 strategic centers.
    # We use a product of 1D Gaussians for efficiency and stability.
    # Centers for t: [30, 40, 50]
    # Centers for x: [-0.2, 0.0, 0.2]
    # Centers for v: [-0.34, 0.0, 0.36]
    
    # We will define 10 basis functions.
    # 3 based on t (low, mid, high)
    # 3 based on x (low, mid, high)
    # 3 based on v (low, mid, high)
    # 1 constant-like (wide Gaussian)
    
    # Widths (sigma) chosen to cover the range without excessive overlap.
    # t range is 20. Sigma_t ~ 5-10.
    # x range is 0.4. Sigma_x ~ 0.1-0.2.
    # v range is 0.7. Sigma_v ~ 0.1-0.2.
    
    # Pre-computed centers and sigmas for the 10 basis functions.
    # Format: (center_t, sigma_t, center_x, sigma_x, center_v, sigma_v)
    
    centers = [
        # 1-3: Time-dependent features (capturing driving/damping evolution)
        (30.0, 8.0, 0.0, 0.2, 0.0, 0.2),
        (40.0, 8.0, 0.0, 0.2, 0.0, 0.2),
        (50.0, 8.0, 0.0, 0.2, 0.0, 0.2),
        
        # 4-6: Position-dependent features (spring force nonlinearity)
        (40.0, 10.0, -0.2, 0.1, 0.0, 0.2),
        (40.0, 10.0,  0.0, 0.1, 0.0, 0.2),
        (40.0, 10.0,  0.2, 0.1, 0.0, 0.2),
        
        # 7-9: Velocity-dependent features (damping nonlinearity)
        (40.0, 10.0,  0.0, 0.2, -0.34, 0.1),
        (40.0, 10.0,  0.0, 0.2,  0.0, 0.1),
        (40.0, 10.0,  0.0, 0.2,  0.36, 0.1),
        
        # 10: Global bias/offset
        (40.0, 20.0,  0.0, 0.4,  0.0, 0.4),
    ]
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate basis functions
    # Using vectorized operations for speed
    
    # Basis 1-3: Time
    b0 = np.exp(-((t - 30.0)**2) / (2 * 8.0**2)) * np.exp(-(x**2) / (2 * 0.2**2)) * np.exp(-(v**2) / (2 * 0.2**2))
    b1 = np.exp(-((t - 40.0)**2) / (2 * 8.0**2)) * np.exp(-(x**2) / (2 * 0.2**2)) * np.exp(-(v**2) / (2 * 0.2**2))
    b2 = np.exp(-((t - 50.0)**2) / (2 * 8.0**2)) * np.exp(-(x**2) / (2 * 0.2**2)) * np.exp(-(v**2) / (2 * 0.2**2))
    
    # Basis 4-6: Position
    b3 = np.exp(-((t - 40.0)**2) / (2 * 10.0**2)) * np.exp(-((x + 0.2)**2) / (2 * 0.1**2)) * np.exp(-(v**2) / (2 * 0.2**2))
    b4 = np.exp(-((t - 40.0)**2) / (2 * 10.0**2)) * np.exp(-(x**2) / (2 * 0.1**2)) * np.exp(-(v**2) / (2 * 0.2**2))
    b5 = np.exp(-((t - 40.0)**2) / (2 * 10.0**2)) * np.exp(-((x - 0.2)**2) / (2 * 0.1**2)) * np.exp(-(v**2) / (2 * 0.2**2))
    
    # Basis 7-9: Velocity
    b6 = np.exp(-((t - 40.0)**2) / (2 * 10.0**2)) * np.exp(-(x**2) / (2 * 0.2**2)) * np.exp(-((v + 0.34)**2) / (2 * 0.1**2))
    b7 = np.exp(-((t - 40.0)**2) / (2 * 10.0**2)) * np.exp(-(x**2) / (2 * 0.2**2)) * np.exp(-(v**2) / (2 * 0.1**2))
    b8 = np.exp(-((t - 40.0)**2) / (2 * 10.0**2)) * np.exp(-(x**2) / (2 * 0.2**2)) * np.exp(-((v - 0.36)**2) / (2 * 0.1**2))
    
    # Basis 10: Global
    b9 = np.exp(-((t - 40.0)**2) / (2 * 20.0**2)) * np.exp(-(x**2) / (2 * 0.4**2)) * np.exp(-(v**2) / (2 * 0.4**2))
    
    # Combine
    result = (
        p0 * b0
        + p1 * b1
        + p2 * b2
        + p3 * b3
        + p4 * b4
        + p5 * b5
        + p6 * b6
        + p7 * b7
        + p8 * b8
        + p9 * b9
    )
    
    return result
# EVOLVE-BLOCK-END
