"""Gaussian Process Regression (Kernel) formulation for symbolic regression."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a sum of Gaussian (squared-exponential) kernels.
    
    The 10 parameters are the weights for 10 fixed Gaussian centers evenly spaced
    across the domain [0.1, 6]. The kernel width is fixed to provide local smoothing.
    
    params: array of length 10, weights for each Gaussian kernel.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Domain is [0.1, 6]
    x_min = 0.1
    x_max = 6.0
    span = x_max - x_min  # 5.9
    
    # 10 evenly spaced centers
    n_centers = 10
    centers = np.linspace(x_min + span / (2 * n_centers), 
                          x_max - span / (2 * n_centers), 
                          n_centers)
    
    # Fixed kernel width (bandwidth) for Gaussian kernels
    # Chosen to be about 1/10 of the domain span for local adaptation
    width = span / 10.0
    
    # Compute the sum of weighted Gaussian kernels
    # For each x, compute sum_i params[i] * exp(-0.5 * ((x - centers[i]) / width)^2)
    
    # Efficient computation using broadcasting
    # x has shape (n_points,), centers has shape (10,)
    # diff has shape (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian values
    # g has shape (n_points, 10)
    g = np.exp(-0.5 * (diff / width) ** 2)
    
    # Combine with parameters (dot product along axis 1)
    # result has shape (n_points,)
    result = g @ params
    
    return result
