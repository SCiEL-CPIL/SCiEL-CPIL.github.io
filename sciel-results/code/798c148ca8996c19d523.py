import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 8 vertices of the domain [0.5, 2]
    # We use hardcoded centers to avoid using parameters for them, 
    # ensuring the optimization landscape is smooth (linear in weights).
    
    # Centers
    c0 = np.array([0.5, 0.5, 0.5])
    c1 = np.array([0.5, 0.5, 2.0])
    c2 = np.array([0.5, 2.0, 0.5])
    c3 = np.array([0.5, 2.0, 2.0])
    c4 = np.array([2.0, 0.5, 0.5])
    c5 = np.array([2.0, 0.5, 2.0])
    c6 = np.array([2.0, 2.0, 0.5])
    c7 = np.array([2.0, 2.0, 2.0])
    
    # Stack centers into (8, 3) array
    centers = np.vstack([c0, c1, c2, c3, c4, c5, c6, c7])
    
    # Inputs into (N, 3) array
    X = np.column_stack((x, y, z))
    
    # Compute squared distances from each point to each center
    # dist_sq[i, j] = || X[i] - centers[j] ||^2
    # Using broadcasting: X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # RBF width sigma
    sigma = 0.9
    # Kernel values: exp(-dist_sq / (2 * sigma^2))
    # Shape: (N, 8)
    K = np.exp(-dist_sq / (2 * sigma**2))
    
    # Parameters:
    # params[0:8] -> weights for the 8 RBFs
    # params[8]   -> bias
    # params[9]   -> coefficient for linear term (x+y+z)
    
    w = params[0:8]
    b = params[8]
    k_lin = params[9]
    
    # Prediction: sum(w_i * K_i) + b + k_lin * (x+y+z)
    pred = K @ w + b + k_lin * (x + y + z)
    
    return pred
