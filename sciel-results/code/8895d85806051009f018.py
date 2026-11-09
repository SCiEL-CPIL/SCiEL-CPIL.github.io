"""
Strategy: Adaptive Gaussian Mixture Basis.

The current program uses a fixed linear combination of global basis functions (polynomials, trig, etc.).
This approach is limited because:
1. Global basis functions are highly correlated, leading to ill-conditioned Hessian matrices for BFGS.
2. The functional form is rigid; if the target function has sharp local features, global polynomials struggle.

This new strategy uses a Gaussian Mixture Model (GMM) with 5 components.
We have 10 parameters total:
- params[0:5] are the centers (mu_0 to mu_4)
- params[5:10] are the inverse variances (sigma_0^2 to sigma_4^2)

The amplitude of each Gaussian is fixed at 1.0. 
Why fixed amplitude?
- It reduces the parameter space dimensionality, making the optimization landscape simpler.
- BFGS starting from all-ones will initially have centers at 1.0 and inverse variances at 1.0.
- The optimizer can move centers to fit local features and adjust widths to smooth or sharpen them.
- Since the amplitudes are fixed, the model is less prone to "collapsing" one Gaussian to fit all data,
  encouraging the 5 components to specialize in different regions of the domain [0.1, 6].

To ensure robustness:
- We clip centers to [0.1, 6] to keep them within the domain.
- We clip inverse variances to [1e-4, 1e4] to prevent division by zero or vanishing Gaussians.
- We use a stable log-sum-exp or direct exp with clipping to avoid overflow.
"""

import numpy as np
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Deadline guard: 150s limit, keep safety margin of 5s
START_TIME = time.time()
DEADLINE = START_TIME + 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute the prediction using a 5-component Gaussian Mixture Model.
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of shape (10,)
        - params[0:5] -> Centers (mu)
        - params[5:10] -> Inverse Variances (sigma^2)
    
    Returns:
    pred: Array of shape (N,) with predictions.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    # Centers: mu_i
    mus = params[0:5]
    # Inverse variances: sigma_i^2
    inv_sigmas_sq = params[5:10]
    
    # Clip centers to the domain [0.1, 6] to ensure relevance
    mus = np.clip(mus, 0.1, 6.0)
    
    # Clip inverse variances to prevent numerical issues
    # If inv_sigma_sq is too small, sigma is huge (flat Gaussian)
    # If inv_sigma_sq is too large, sigma is tiny (spike)
    # We want to allow a range of widths.
    # Let's clip inv_sigmas_sq to [1e-4, 100.0]
    inv_sigmas_sq = np.clip(inv_sigmas_sq, 1e-4, 100.0)
    
    # Initialize prediction
    pred = np.zeros_like(x)
    
    # Loop over the 5 Gaussian components
    # This is fast enough for 200-400 points
    for i in range(5):
        mu = mus[i]
        inv_sig2 = inv_sigmas_sq[i]
        
        # Compute Gaussian: exp(-0.5 * (x - mu)^2 * inv_sig2)
        # To avoid overflow in (x-mu)^2 * inv_sig2, we can clip the exponent
        diff = x - mu
        exponent = -0.5 * diff * diff * inv_sig2
        
        # Clip exponent to [-700, 0] to prevent underflow/overflow in exp
        # exp(-700) is approx 1e-304, which is fine for float64
        # We don't need positive exponent clipping because exponent is always <= 0
        exponent = np.clip(exponent, -700.0, 0.0)
        
        gauss = np.exp(exponent)
        
        # Amplitude is fixed at 1.0
        pred += 1.0 * gauss
        
    return pred

# EVOLVE-BLOCK-END
