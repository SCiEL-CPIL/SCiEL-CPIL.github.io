# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a diverse, normalized basis.
    
    The basis includes polynomials, trigonometric, and hyperbolic functions.
    Each basis function is normalized (zero mean, unit variance) to improve
    the condition number of the design matrix for BFGS optimization.
    """
    # 1. Define raw basis functions
    # We choose functions likely to appear in symbolic regression benchmarks
    # and ensure they are smooth and well-behaved on [-2, 2].
    
    f0 = np.ones_like(x)
    f1 = x
    f2 = x**2
    f3 = x**3
    f4 = np.sin(x)
    f5 = np.cos(x)
    f6 = np.exp(x / 2.0)  # Scaled to avoid large values
    f7 = np.tanh(x)
    f8 = x / (1.0 + x**2) # Bounded rational function
    f9 = np.sin(2.0 * x)   # Higher frequency trig
    
    basis = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # 2. Normalize basis columns
    # This step is critical for BFGS stability.
    # We compute mean and std for each column.
    # If std is very small (constant function), we leave it as is (or set to 1).
    
    means = np.mean(basis, axis=0)
    stds = np.std(basis, axis=0)
    
    # Avoid division by zero
    stds[stds < 1e-12] = 1.0
    
    # Normalize: (x - mean) / std
    # Note: We store the normalization constants to apply them to the prediction?
    # No, the equation function takes `params` which are fitted by the evaluator.
    # The evaluator minimizes MSE between `equation(x, params)` and `y`.
    # If we normalize the basis inside `equation`, the `params` will be the coefficients
    # of the normalized basis. This is fine. The evaluator just sees a linear map.
    
    basis_norm = (basis - means) / stds
    
    # 3. Compute prediction
    # params is a vector of length 10.
    # prediction = basis_norm @ params
    
    return basis_norm @ params
# EVOLVE-BLOCK-END
