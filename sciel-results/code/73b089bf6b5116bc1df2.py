# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local Linearization with Gaussian RBFs.
    
    Structural change from global polynomial/Fourier to local phase-plane approximation.
    Uses 3 Gaussian centers in (x, v) space.
    Each center is associated with a local linear model: w0 + w1*x + w2*v.
    Total parameters: 3 centers * 3 coeffs = 9 parameters.
    
    This allows the model to adapt to local curvature in the phase plane,
    potentially handling non-additive couplings better than a global basis,
    although it lacks explicit time-dependence (which may be a trade-off).
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    n = len(t)
    
    # Define 3 centers in (x, v) space
    # x range approx [-0.2, 0.2], v range approx [-0.3, 0.3]
    x_centers = np.array([0.0, -0.15, 0.15])
    v_centers = np.array([0.0, -0.20, 0.20])
    
    # Fixed width for Gaussians
    sigma = 0.15
    sigma2 = 2 * sigma * sigma
    
    # Compute RBFs
    # R_k(x, v) = exp(-((x - xc_k)^2 + (v - vc_k)^2) / (2 * sigma^2))
    
    # Center 0
    dx0 = x - x_centers[0]
    dv0 = v - v_centers[0]
    r0 = np.exp(-(dx0*dx0 + dv0*dv0) / sigma2)
    
    # Center 1
    dx1 = x - x_centers[1]
    dv1 = v - v_centers[1]
    r1 = np.exp(-(dx1*dx1 + dv1*dv1) / sigma2)
    
    # Center 2
    dx2 = x - x_centers[2]
    dv2 = v - v_centers[2]
    r2 = np.exp(-(dx2*dx2 + dv2*dv2) / sigma2)
    
    # Local linear terms
    # For each center, we have a linear function: p0 + p1*x + p2*v
    
    # Params 0, 1, 2 for Center 0
    lin0 = params[0] + params[1] * x + params[2] * v
    
    # Params 3, 4, 5 for Center 1
    lin1 = params[3] + params[4] * x + params[5] * v
    
    # Params 6, 7, 8 for Center 2
    lin2 = params[6] + params[7] * x + params[8] * v
    
    # Parameter 9 is unused (or can be used for a global bias if needed, but 9 is enough)
    # We have 10 slots. Let's add a global constant bias to help fitting?
    # Or just leave it. The local biases cover it.
    
    # Combine
    pred = r0 * lin0 + r1 * lin1 + r2 * lin2
    
    return pred
# EVOLVE-BLOCK-END
