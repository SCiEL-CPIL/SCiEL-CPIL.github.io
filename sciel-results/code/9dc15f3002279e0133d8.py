"""Symbolic regression with RBF basis and robust fitting."""
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a linear combination of 10 RBFs with a single learnable width.
    params[0] = width parameter (log scale for stability)
    params[1:10] = weights for the 10 RBF centers
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # 10 evenly spaced centers in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Width from params[0], using exp to ensure positive and smooth
    # Start at 1.0, so width = exp(0) = 1.0
    log_width = params[0]
    width = np.exp(log_width)
    
    # Compute RBF values: exp(-((x - c)^2) / (2 * width^2))
    # x shape: (n,), centers shape: (10,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    rbf_vals = np.exp(-diff**2 / (2.0 * width**2))  # (n, 10)
    
    # Weights are params[1:10]
    weights = params[1:10]  # (10,)
    
    # Prediction: rbf_vals @ weights
    pred = rbf_vals @ weights
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred

# Pre-compute centers for efficiency (though they're simple enough)
_CENTERS = np.linspace(-3.0, 3.0, 10)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a linear combination of 10 RBFs with a single learnable width.
    params[0] = log-width parameter
    params[1:10] = weights for the 10 RBF centers
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Width from params[0]
    log_width = params[0]
    # Clamp log_width to avoid numerical issues
    log_width = np.clip(log_width, -3.0, 3.0)  # width in [exp(-3), exp(3)] ~ [0.05, 20]
    width = np.exp(log_width)
    
    # Compute RBF values
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    rbf_vals = np.exp(-diff**2 / (2.0 * width**2))
    
    # Weights
    weights = params[1:10]
    
    pred = rbf_vals @ weights
    
    # Safety: replace non-finite with 0
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
