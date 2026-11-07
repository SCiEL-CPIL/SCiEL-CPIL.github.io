"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a linear combination of 10 RBFs
    at fixed, evenly spaced centers on [-3, 3], with a single learnable width.
    
    params: array of length 10
      - params[0:9]: weights for the 10 RBFs (we use 9 weights + 1 width = 10 params)
      - params[9]: inverse of the width (log-scale for stability)
    
    Wait, the task says params has length 10. We need 10 weights for 10 RBFs + 1 width = 11 params.
    But we only have 10. Let's use 9 RBFs + 1 width = 10 params.
    Or 10 RBFs with a fixed width? The directive says "single learnable width parameter shared by all kernels".
    
    Let's use 9 RBFs and 1 width parameter. That gives 10 params total.
    Centers: evenly spaced in [-3, 3]. With 9 centers: -3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3.
    
    Actually, let's reconsider. The directive says "10 RBFs at fixed, evenly spaced centers, using a single learnable width parameter". 
    That would be 10 + 1 = 11 parameters. But we only have 10.
    
    Option 1: Use 9 RBFs + 1 width = 10 params.
    Option 2: Use 10 RBFs with fixed width (e.g., 1.0) = 10 params.
    Option 3: Use 10 RBFs, but the width is derived from one of the weights (hacky).
    
    Let's go with Option 1: 9 RBFs + 1 width. This satisfies "single learnable width" and uses exactly 10 params.
    The centers will be 9 evenly spaced points in [-3, 3].
    
    RBF: phi(x, c, sigma) = exp(-((x - c)^2) / (2 * sigma^2))
    We use log_sigma = params[9], so sigma = exp(log_sigma).
    To keep sigma in a reasonable range, we can constrain it or just let BFGS handle it.
    Starting from all ones: log_sigma = 1 -> sigma = e ≈ 2.718. That's a wide kernel.
    Maybe we should start with a different initial guess? The evaluator starts from all ones.
    If sigma = e, the RBFs are very wide and might be nearly linear. That's fine for BFGS to start.
    
    Let's define:
    centers = np.linspace(-3, 3, 9)
    weights = params[0:9]
    log_sigma = params[9]
    sigma = np.exp(log_sigma)
    
    To avoid overflow/underflow in exp, we can clip log_sigma.
    sigma = np.exp(np.clip(log_sigma, -5, 5))
    
    Prediction: sum(weights[i] * exp(-((x - centers[i])^2) / (2 * sigma^2)) for i in range(9))
    
    This is a smooth, well-conditioned approximation. BFGS should handle it well.
    """
    start_time = time.time()
    
    # Define fixed centers for 9 RBFs
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Extract parameters
    weights = params[0:9]
    log_sigma = params[9]
    
    # Compute sigma with clipping for numerical stability
    log_sigma_clipped = np.clip(log_sigma, -5.0, 5.0)
    sigma = np.exp(log_sigma_clipped)
    sigma2 = sigma * sigma
    
    # Compute RBF values: shape (n_points, 9)
    # x has shape (n_points,)
    # centers has shape (9,)
    # diff = x[:, np.newaxis] - centers[np.newaxis, :] -> shape (n_points, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-diff**2 / (2.0 * sigma2))
    
    # Prediction: rbf_values @ weights -> shape (n_points,)
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
