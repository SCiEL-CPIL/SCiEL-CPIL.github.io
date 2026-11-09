"""
Strategy: Fixed-Center Gaussian Radial Basis Function (RBF) Regression.

Rationale:
1. The previous strategy used a monomial polynomial basis of degree 9. While linear in parameters,
   the monomial basis $\{1, x, ..., x^9\}$ is notoriously ill-conditioned for $x \in [-3, 3]$.
   The condition number of the Vandermonde matrix grows rapidly, which can cause BFGS to struggle
   with numerical precision, leading to slow convergence or failure to find the optimal coefficients,
   especially if the target function has sharp local variations that require high-frequency components.
2. This strategy uses a Gaussian RBF basis with 10 fixed centers evenly spaced in $[-3, 3]$.
   The model is $f(x) = \sum_{i=0}^{9} w_i \exp(-\frac{(x - c_i)^2}{2\sigma^2})$.
3. Key advantages:
   a) Linearity in parameters: The optimization remains convex with respect to the weights $w_i$,
      ensuring BFGS will converge to the global minimum for the fixed basis.
   b) Numerical Stability: The RBF matrix is typically much better conditioned than the high-degree
      monomial Vandermonde matrix for this domain size. The values are bounded and decay smoothly.
   c) Local Flexibility: Unlike global polynomials, RBFs allow the model to fit local features
      (peaks, valleys, changes in slope) independently at different locations in the domain,
      avoiding Runge's phenomenon oscillations.
   d) Fixed Width: We choose a fixed $\sigma$ (e.g., 0.8) that provides sufficient overlap between
      neighbors to ensure the system is not under-determined or too sparse, while still allowing
      local adaptation. This avoids the non-convexity of optimizing widths.
4. The 10 parameters correspond directly to the weights of the 10 basis functions.
"""

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a fixed-center Gaussian RBF basis.
    
    params: array of 10 weights [w_0, ..., w_9].
    x: input array of shape (N,).
    
    Returns: array of shape (N,) with predictions.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 fixed centers evenly spaced in [-3, 3]
    # np.linspace(-3, 3, 10) gives centers at -3.0, -2.333, ..., 3.0
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width (sigma). 
    # A sigma of 0.8 ensures significant overlap between adjacent centers (distance ~0.67),
    # making the basis functions linearly independent enough for stable regression 
    # while retaining locality.
    sigma = 0.8
    sigma2 = 2.0 * sigma * sigma
    
    # Compute the basis matrix B of shape (N, 10)
    # B[i, j] = exp(-0.5 * (x_i - c_j)^2 / sigma^2)
    # We compute this efficiently using broadcasting.
    # x shape: (N, 1)
    # centers shape: (1, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    sq_diff = diff * diff
    exponents = -0.5 * sq_diff / sigma2
    basis = np.exp(exponents)
    
    # Compute prediction: y = B @ params
    # basis shape: (N, 10)
    # params shape: (10,)
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
