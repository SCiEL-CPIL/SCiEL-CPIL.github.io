"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10 RBFs at fixed, evenly spaced centers in [-3, 3]
    # Centers: -3, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4
    centers = np.array([-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4])
    
    # params[0] is the learnable width (sigma) for all RBFs
    # params[1:10] are the weights for the 9 RBFs (we use 9 weights + 1 width = 10 params)
    # Actually, let's use all 10 params: 1 width + 9 weights? Or 1 width + 10 weights would need 11 params.
    # The task says params is an array of length 10.
    # Let's use params[0] as width (sigma), and params[1:10] as 9 weights.
    # But we have 10 centers. Let's reconsider.
    
    # Option: Use 10 RBFs. params[0] = sigma, params[1:10] = 9 weights. One center is unused?
    # Better: Use 9 RBFs. Centers: -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4
    # Or: params[0] = sigma, params[1:10] = 9 weights for 9 centers.
    
    # Let's use 9 centers to fit 9 weights + 1 width = 10 params.
    centers = np.array([-2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4])
    
    sigma = np.abs(params[0]) + 0.1  # Ensure positive sigma, minimum 0.1
    weights = params[1:10]
    
    # Compute RBF values: exp(-0.5 * ((x - center) / sigma)^2)
    # x shape: (N,), centers shape: (9,)
    # We need to compute (N, 9) matrix of RBF values
    
    x_expanded = x[:, np.newaxis]  # (N, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 9)
    
    # Compute squared distances
    diff = x_expanded - centers_expanded  # (N, 9)
    sq_dist = diff ** 2
    
    # RBF values
    rbf_values = np.exp(-0.5 * sq_dist / (sigma ** 2))  # (N, 9)
    
    # Prediction: sum of weights * rbf_values
    prediction = rbf_values @ weights  # (N,)
    
    return prediction
# EVOLVE-BLOCK-END
