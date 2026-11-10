import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    if p.size != 10:
        # Fallback: try to reshape or pad
        if p.size > 10:
            p = p[:10]
        else:
            p = np.pad(p, (0, 10 - p.size))
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Model: y(x) = sum_{i=0}^{4} A_i * exp(-((x - mu_i)^2) / (2 * sigma_i^2))
    # Parameters:
    # p[0:5] -> Amplitudes A_i
    # p[5:10] -> Means mu_i
    # Widths sigma_i are fixed to 0.5 to maintain linearity in A_i and mu_i is non-linear but local.
    # Actually, to make it more robust and potentially capture broader shapes, 
    # let's use a mix of Gaussian RBFs.
    # To keep it simple and linear in amplitudes, we fix sigmas.
    # However, BFGS can handle non-linear parameters.
    # Let's use:
    # A_i = p[i]
    # mu_i = p[5+i]
    # sigma_i = 0.5 (fixed)
    
    A = p[:5]
    mu = p[5:10]
    
    # Fixed widths
    sigma = 0.5
    sigma_sq = 2.0 * sigma * sigma
    
    y = np.zeros_like(x)
    
    for i in range(5):
        # Compute Gaussian
        diff = x - mu[i]
        # Clip diff to prevent overflow in exp, though x in [-2,2] and mu likely similar
        # exp(-diff^2 / sigma_sq) is always <= 1
        y += A[i] * np.exp(-(diff * diff) / sigma_sq)
        
    # Clip output to prevent extreme values that might cause issues in MSE calculation
    # Although exp is bounded, A_i can be large.
    y = np.clip(y, -1e150, 1e150)
    
    return y
