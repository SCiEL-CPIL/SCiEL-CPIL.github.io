import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 1. Define 5 fixed centers evenly spaced across the input domain [-3, 3]
    # Centers: -3, -1.5, 0, 1.5, 3
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # 2. Extract parameters
    # params[0:5] -> Weights for the RBFs
    # params[5:10] -> Log-widths for the RBFs
    weights = params[0:5]
    log_widths = params[5:10]
    
    # 3. Compute widths
    # Using exp(log_width) ensures positive widths and improves BFGS conditioning.
    # Initial log_width = 1.0 -> width = e^1 ≈ 2.7, which covers the domain well.
    widths = np.exp(log_widths)
    
    # 4. Compute RBF values
    # x is (N,), centers is (5,)
    # We want a matrix D of shape (N, 5) where D[i, j] = exp(-0.5 * ((x_i - c_j) / width_j)^2)
    
    # x[:, np.newaxis] is (N, 1)
    # centers[np.newaxis, :] is (1, 5)
    # widths[np.newaxis, :] is (1, 5)
    
    # Avoid division by zero if width becomes very small, though exp(log_w) is always > 0.
    # However, if log_w is very negative, width is tiny, and (x-c)/w can be huge, leading to overflow in square or underflow in exp.
    # Underflow in exp is fine (goes to 0). Overflow in square might be an issue if x-c is large and w is tiny.
    # x in [-3,3], c in [-3,3], so x-c in [-6,6].
    # If w is 1e-10, (x-c)/w is 6e10. Squared is 3.6e21. exp(-0.5 * 3.6e21) is 0.
    # NumPy handles this gracefully (returns 0.0).
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    scaled_diff = diff / widths[np.newaxis, :]
    
    # Compute squared distance
    # Use np.square to be explicit
    squared_dist = np.square(scaled_diff)
    
    # Compute Gaussian
    # Clip the argument to prevent any potential issues, though exp(-large) is stable.
    # The main risk is overflow in squared_dist if widths are extremely small.
    # Let's add a small epsilon to widths to be safe against 0 division if log_w -> -inf (unlikely with BFGS but possible).
    # Actually, exp(log_w) with log_w = -1000 is 0. Division by 0 gives inf. inf^2 is inf. exp(-inf) is 0.
    # This is mathematically correct for a Dirac delta limit.
    # However, 0/0 is NaN. If x==c and w==0, diff is 0. 0/0 is NaN.
    # To prevent NaNs, we can add a tiny epsilon to widths.
    eps = 1e-12
    widths_safe = widths + eps
    
    scaled_diff_safe = diff / widths_safe[np.newaxis, :]
    squared_dist_safe = np.square(scaled_diff_safe)
    
    # Calculate RBF values
    rbf_values = np.exp(-0.5 * squared_dist_safe)
    
    # 5. Combine with weights
    # rbf_values is (N, 5), weights is (5,)
    # y = rbf_values @ weights
    y = rbf_values @ weights
    
    return y
