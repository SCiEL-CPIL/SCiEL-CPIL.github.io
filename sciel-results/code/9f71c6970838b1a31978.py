"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian Radial Basis Function (RBF) expansion.
    
    This approach uses a fixed grid of 9 RBF centers on the domain [-1, 1] x [-1, 1]
    and 1 bias term. The parameters correspond to the weights of these basis functions.
    RBFs are flexible and can capture local variations better than low-degree polynomials.
    """
    
    # Define 9 centers in a 3x3 grid
    centers = [
        (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
        (0.0, -1.0),  (0.0, 0.0),  (0.0, 1.0),
        (1.0, -1.0),  (1.0, 0.0),  (1.0, 1.0)
    ]
    
    # Set the width (sigma) for the Gaussian kernels
    # A width of 0.6 provides a good balance between local flexibility and smoothness
    sigma = 0.6
    sigma2 = 2.0 * sigma**2
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Compute the RBF terms
    # We calculate the squared distance from each point to each center
    # r_i^2 = (x - cx_i)^2 + (y - cy_i)^2
    # term_i = exp(-r_i^2 / sigma2)
    
    # To optimize performance, we compute all terms vectorized
    # x and y are arrays of shape (N,)
    
    # Term 0: Bias (or constant RBF at center 0? No, let's use pure RBFs + bias)
    # Let's assign:
    # params[0] is the weight for the bias term (1)
    # params[1] to params[9] are weights for the 9 RBFs
    
    # Wait, the previous code used p0 as bias. Let's stick to that convention.
    # p0: bias
    # p1-p9: RBF weights
    
    rbf_terms = np.zeros_like(x)
    
    # Pre-compute x^2 and y^2 if needed, but direct subtraction is fine
    # Center 0: (-1, -1)
    rbf_terms += p1 * np.exp(-((x - (-1.0))**2 + (y - (-1.0))**2) / sigma2)
    # Center 1: (-1, 0)
    rbf_terms += p2 * np.exp(-((x - (-1.0))**2 + (y - 0.0)**2) / sigma2)
    # Center 2: (-1, 1)
    rbf_terms += p3 * np.exp(-((x - (-1.0))**2 + (y - 1.0)**2) / sigma2)
    # Center 3: (0, -1)
    rbf_terms += p4 * np.exp(-((x - 0.0)**2 + (y - (-1.0))**2) / sigma2)
    # Center 4: (0, 0)
    rbf_terms += p5 * np.exp(-((x - 0.0)**2 + (y - 0.0)**2) / sigma2)
    # Center 5: (0, 1)
    rbf_terms += p6 * np.exp(-((x - 0.0)**2 + (y - 1.0)**2) / sigma2)
    # Center 6: (1, -1)
    rbf_terms += p7 * np.exp(-((x - 1.0)**2 + (y - (-1.0))**2) / sigma2)
    # Center 7: (1, 0)
    rbf_terms += p8 * np.exp(-((x - 1.0)**2 + (y - 0.0)**2) / sigma2)
    # Center 8: (1, 1)
    rbf_terms += p9 * np.exp(-((x - 1.0)**2 + (y - 1.0)**2) / sigma2)
    
    return p0 + rbf_terms
# EVOLVE-BLOCK-END
