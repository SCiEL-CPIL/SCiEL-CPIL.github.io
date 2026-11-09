import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    t_start = time.time()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # 5 evenly spaced centers across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract weights (first 5) and log-widths (next 5)
    weights = params[:5]
    log_widths = params[5:10]
    
    # Widths: exp(log_width) gives initial sigma ~ e^1 = 2.718 when params start at 1
    # This is well-suited for domain [-3, 3]
    widths = np.exp(log_widths)
    
    # Compute RBF basis: 5 Gaussian basis functions
    # For each center c_i and width sigma_i, basis_i(x) = exp(-((x - c_i)^2) / (2 * sigma_i^2))
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(5):
        c = centers[i]
        sigma = widths[i]
        # Guard against zero or negative width
        sigma = max(sigma, 1e-6)
        diff = x - c
        basis_val = np.exp(-0.5 * (diff / sigma) ** 2)
        result += weights[i] * basis_val
    
    return result
