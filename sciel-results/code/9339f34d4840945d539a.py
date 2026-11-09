import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Mixture of Gaussian Radial Basis Functions (RBFs).
    
    The model is defined as:
    y = w_0 + w_1 * x + sum_{i=1}^{4} a_i * exp(-((x - c_i)^2) / (2 * sigma_i^2))
    
    Parameters (length 10):
    p[0]: w_0 (bias)
    p[1]: w_1 (linear coefficient)
    p[2]: a_1 (amplitude of RBF 1)
    p[3]: c_1 (center of RBF 1)
    p[4]: log_sigma_1 (log width of RBF 1, ensures sigma > 0)
    p[5]: a_2 (amplitude of RBF 2)
    p[6]: c_2 (center of RBF 2)
    p[7]: log_sigma_2 (log width of RBF 2)
    p[8]: a_3 (amplitude of RBF 3)
    p[9]: c_3 (center of RBF 3)
    
    Note: To fit 10 params, I will use 3 RBFs. 
    Wait, 3 RBFs * 3 params (a, c, sigma) = 9 params. Plus 1 bias = 10.
    I will drop the linear term w_1*x to keep it simpler and more local, or keep it and drop one RBF?
    Let's use 3 RBFs with learnable centers, widths, and amplitudes.
    p[0]: a_1
    p[1]: c_1
    p[2]: log_sigma_1
    p[3]: a_2
    p[4]: c_2
    p[5]: log_sigma_2
    p[6]: a_3
    p[7]: c_3
    p[8]: log_sigma_3
    p[9]: bias
    
    This covers the whole domain with 3 flexible bumps.
    """
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Extract parameters
    a1, c1, log_s1 = params[0], params[1], params[2]
    a2, c2, log_s2 = params[3], params[4], params[5]
    a3, c3, log_s3 = params[6], params[7], params[8]
    bias = params[9]
    
    # Ensure widths are positive and reasonably bounded to avoid numerical issues
    # sigma = exp(log_sigma). If log_sigma is very large, exp overflows.
    # If log_sigma is very small, exp underflows to 0, causing division by zero in exp(-x/0).
    # Clip log_sigma to [-10, 10] to keep sigma in [~0, ~22000]
    # Actually, for x in [-3, 3], a width of 10 is huge (flat), width 0.01 is very sharp.
    # Let's clip to [-5, 5] -> sigma in [0.006, 148].
    s1 = np.exp(np.clip(log_s1, -5, 5))
    s2 = np.exp(np.clip(log_s2, -5, 5))
    s3 = np.exp(np.clip(log_s3, -5, 5))
    
    # Compute RBFs
    # Gaussian: exp(-((x - c)^2) / (2 * s^2))
    # To prevent underflow/overflow in the exponent, we can just compute directly.
    # (x-c) is max 6. (x-c)^2 is max 36.
    # If s is small, 2*s^2 is small, exponent is large negative -> exp -> 0. Safe.
    # If s is large, exponent is small negative -> exp -> 1. Safe.
    
    term1 = a1 * np.exp(-((x - c1) ** 2) / (2.0 * s1 ** 2))
    term2 = a2 * np.exp(-((x - c2) ** 2) / (2.0 * s2 ** 2))
    term3 = a3 * np.exp(-((x - c3) ** 2) / (2.0 * s3 ** 2))
    
    y = bias + term1 + term2 + term3
    
    return y
