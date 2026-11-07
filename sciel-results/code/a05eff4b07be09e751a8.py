import numpy as np

def heuristics(demands, capacity):
    demands = np.array(demands, dtype=float)
    n = len(demands)
    
    # Calculate pair sums
    sums = demands[:, None] + demands[None, :]
    residuals = capacity - sums
    
    # Validity mask
    valid = residuals >= 0
    
    # Base heuristic: inverse residual
    # Add 1.0 to avoid division by zero and smooth
    base = np.zeros((n, n), dtype=float)
    base[valid] = 1.0 / (residuals[valid] + 1.0)
    
    # Size boost: prefer placing larger items (j)
    # demands[None, :] broadcasts row-wise (j index)
    boost = 1.0 + demands[None, :] / capacity
    
    # Combine
    eta = base * boost
    
    # Ensure non-positive become 1e-9
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
