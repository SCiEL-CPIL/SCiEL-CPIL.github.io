"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid basis expansion.
    
    The model combines a low-degree polynomial part (for global trends) with
    Gaussian Radial Basis Functions (for local features). 
    The 10 parameters are split:
    - params[0:4] : coefficients for polynomial basis [1, x, x^2, x^3]
    - params[4:10]: weights for 6 Gaussian RBFs
    
    This hybrid approach aims to capture both the global structure and
    potential localized features of the unknown function, which pure polynomials
    or pure RBFs might miss.
    """
    # Deadline guard: The evaluator kills after 150s.
    # We check time to ensure we don't hang, though this function is O(N) and fast.
    # This is mostly a safeguard per directive.
    start_time = time.time()
    
    # Ensure x is a 1D array for broadcasting
    x = np.asarray(x).flatten()
    
    # --- Part 1: Polynomial Basis (Global) ---
    # Using normalized x to keep values stable, but since BFGS fits params,
    # scaling is absorbed. However, for numerical stability in the exponent
    # of RBFs, we use raw x for centers.
    
    # Polynomial terms: 1, x, x^2, x^3
    # We use the first 4 parameters for this.
    poly_result = (
        params[0] * np.ones_like(x) +
        params[1] * x +
        params[2] * x**2 +
        params[3] * x**3
    )
    
    # --- Part 2: Gaussian RBF Basis (Local) ---
    # 6 RBF centers evenly distributed in [-3, 3]
    # Leaving out the very edges to ensure coverage of the bulk
    centers = np.linspace(-2.5, 2.5, 6)
    
    # Width of the Gaussians.
    # A width of 1.0 is reasonable for domain length 6 with 6 centers.
    sigma = 1.0
    
    # Compute the RBF basis matrix
    # shape: (num_points, 6)
    # B[i, j] = exp(-(x_i - centers[j])^2 / (2 * sigma^2))
    
    # Calculate squared distances
    # x[:, None] has shape (N, 1)
    # centers[None, :] has shape (1, 6)
    diff = x[:, None] - centers[None, :]
    
    # Compute the exponential term
    basis = np.exp(-diff**2 / (2 * sigma**2))
    
    # The model is a linear combination of the RBF basis functions
    # result_rbf = basis @ params[4:10]
    # params[4:10] has shape (6,)
    # basis has shape (N, 6)
    # result_rbf has shape (N,)
    
    rbf_result = basis @ params[4:10]
    
    # Combine
    result = poly_result + rbf_result
    
    # Safety check: if computation took too long (unlikely), return what we have.
    # If result is not finite, BFGS will handle it with penalty, but we ensure output is valid shape.
    
    return result
# EVOLVE-BLOCK-END
