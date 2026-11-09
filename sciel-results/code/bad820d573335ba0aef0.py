import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a sum of 3 localized Gaussian RBFs + linear trend.
    Parameters:
    p[0:3]: Linear coefficients for x, y, z
    p[3:6]: Centers (cx, cy, cz) for RBF 1
    p[6:9]: Centers (cx, cy, cz) for RBF 2
    p[9]:   Amplitude for RBF 1 (RBF 2 amplitude fixed to 1.0 for stability)
    
    Note: To maximize flexibility within 10 params, we use:
    p[0:3]: Linear coeffs
    p[3:6]: Center 1 (cx1, cy1, cz1)
    p[6:9]: Center 2 (cx2, cy2, cz2)
    p[9]:   Width parameter (shared for both RBFs to reduce non-convexity issues)
    
    Actually, let's use a more standard decomposition:
    p[0]: Intercept
    p[1:4]: Linear coeffs (x, y, z)
    p[4:7]: Center 1 (cx1, cy1, cz1)
    p[7:10]: Center 2 (cx2, cy2, cz2)
    
    Wait, we need amplitudes and widths too. With only 10 params, we must fix some.
    Let's try:
    p[0:3]: Linear coeffs for x, y, z
    p[3:6]: Center 1 (cx1, cy1, cz1)
    p[6:9]: Center 2 (cx2, cy2, cz2)
    p[9]:   Width sigma (shared)
    
    Amplitudes: Fixed to 1.0? Or maybe p[0] is intercept, and amplitudes are derived?
    If amplitudes are fixed to 1, the model is:
    f = p0*x + p1*y + p2*z + 1.0*exp(-||r-r1||^2 / (2*sigma^2)) + 1.0*exp(-||r-r2||^2 / (2*sigma^2))
    
    This might be too restrictive on amplitude. Let's try to make amplitudes learnable.
    Maybe:
    p[0:3]: Linear coeffs
    p[3:6]: Center 1
    p[6:9]: Center 2
    p[9]:   Width
    
    If we fix amplitudes to 1, we can't scale the bumps. 
    Alternative:
    p[0]: Intercept
    p[1:4]: Linear coeffs
    p[4:7]: Center 1
    p[7:10]: Center 2
    
    And fix widths to a reasonable value like 0.5, and amplitudes to 1?
    No, that's not flexible enough.
    
    Let's look at the inspiration 3: "sum of three independent, localized radial basis functions (one per input variable) plus a linear trend".
    This suggests:
    f = p0 + p1*x + p2*y + p3*z + 
        p4 * exp(-(x-c1)^2 / w^2) + 
        p5 * exp(-(y-c2)^2 / w^2) + 
        p6 * exp(-(z-c3)^2 / w^2)
        
    This uses:
    p0: Intercept
    p1-p3: Linear
    p4: Amp 1
    p5: Amp 2
    p6: Amp 3
    p7: Center 1 (for x)
    p8: Center 2 (for y)
    p9: Center 3 (for z)
    
    This leaves widths fixed. Let's fix widths to 0.5 (half the range of inputs).
    This is a 1D RBF approach, which is less expressive than 3D RBFs but uses the 10 params more efficiently (each param has a clear role).
    """
    
    # Deadline guard
    start_time = time.time()
    
    # Inputs are in [0.5, 2.0]
    # We use the raw inputs directly for the RBF centers and linear terms to avoid scaling issues with BFGS.
    
    p = params
    
    # Linear part
    lin = p[0] + p[1]*x + p[2]*y + p[3]*z
    
    # RBF part
    # Fixed width sigma = 0.5
    sigma = 0.5
    sigma2 = 2 * sigma * sigma
    
    # 1D RBFs for each dimension
    # Centers are p[7], p[8], p[9]
    # Amplitudes are p[4], p[5], p[6]
    
    # To prevent overflow in exp, we clip the exponent
    ex1 = -((x - p[7])**2) / sigma2
    ex2 = -((y - p[8])**2) / sigma2
    ex3 = -((z - p[9])**2) / sigma2
    
    # Clip to avoid overflow (though inputs are bounded, centers might go wild during optimization)
    ex1 = np.clip(ex1, -50, 0)
    ex2 = np.clip(ex2, -50, 0)
    ex3 = np.clip(ex3, -50, 0)
    
    rbf1 = p[4] * np.exp(ex1)
    rbf2 = p[5] * np.exp(ex2)
    rbf3 = p[6] * np.exp(ex3)
    
    return lin + rbf1 + rbf2 + rbf3
