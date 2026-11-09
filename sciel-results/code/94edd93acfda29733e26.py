"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 8 fixed Gaussian centers spanning [0.1, 6].
    params[0]: learnable width (sigma) of the Gaussian kernels.
    params[1:9]: weights for each of the 8 RBF centers.
    
    The centers are fixed at equally spaced points in the domain.
    A small positive offset is added to sigma to keep it positive during optimization.
    """
    # Fixed centers spanning the input domain [0.1, 6]
    centers = np.linspace(0.1, 6.0, 8)
    
    # Extract parameters
    # Use a softplus-like transformation or just clamp to ensure positive sigma
    # Since BFGS starts from all ones, params[0] = 1 initially.
    # We want sigma > 0. Let's use exp(params[0]) to ensure positivity and differentiability.
    # However, if params[0] is large, exp might overflow. 
    # Alternatively, we can use np.maximum(params[0], 1e-6) but that's not differentiable at 0.
    # Let's stick with exp(params[0]) for smoothness, but maybe scale it down?
    # Actually, the domain is [0.1, 6]. A reasonable sigma is around 0.5-1.0.
    # If we use exp(params[0]), starting from 1, sigma = e ~ 2.7. That's quite wide.
    # Maybe use 0.5 * exp(params[0])? Or just params[0] + 0.1?
    # Let's use a simple shift: sigma = params[0] + 0.1. This is differentiable everywhere.
    # If BFGS pushes it negative, it might cause issues if we square it, but we don't.
    # Wait, if sigma is negative, the Gaussian is still valid (symmetric).
    # But typically we want width > 0. Let's use np.abs(params[0]) + 0.1? No, abs is not differentiable at 0.
    # Let's use exp(params[0]) * 0.5. Starting from 1, sigma = 0.5 * e ~ 1.35.
    # This seems reasonable.
    
    sigma = 0.5 * np.exp(params[0])
    
    # Compute distances from x to each center
    # x shape: (N,)
    # centers shape: (8,)
    # diff shape: (N, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian kernels
    # exp(-diff^2 / (2 * sigma^2))
    kernel = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Linear combination of kernels
    # weights shape: (8,)
    weights = params[1:9]
    
    # prediction = kernel @ weights
    prediction = kernel @ weights
    
    return prediction
# EVOLVE-BLOCK-END
