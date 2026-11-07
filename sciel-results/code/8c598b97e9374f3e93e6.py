import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Extract weights and log-width
    weights = params[:9]
    log_sigma = params[9]
    sigma = np.exp(log_sigma)
    
    # Define 10 fixed centers in [-3, 3]
    centers = np.linspace(-3, 3, 10)
    
    # Compute RBFs
    # x shape: (N, 1) or (N,)
    # centers shape: (10,)
    # We want to compute exp(-(x - c_i)^2 / (2 * sigma^2)) for each i
    # x[:, None] has shape (N, 1)
    # centers[None, :] has shape (1, 10)
    # diff has shape (N, 10)
    diff = x[:, None] - centers[None, :]
    
    # Compute squared distance
    sq_dist = diff ** 2
    
    # Compute RBF values
    # sigma^2
    sigma_sq = sigma ** 2
    rbf_values = np.exp(-sq_dist / (2.0 * sigma_sq))
    
    # We have 10 RBFs but only 9 weights. 
    # Let's use the 10th weight as the first weight to satisfy the "10 RBFs" idea 
    # while keeping 10 parameters total.
    # Or simply use 9 RBFs. The directive says "10 RBFs".
    # Let's set weight_9 = weight_0.
    w = np.append(weights, weights[0])
    
    # Linear combination
    y = rbf_values @ w
    
    return y
