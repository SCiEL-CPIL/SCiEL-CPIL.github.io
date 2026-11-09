"""
Strategy: Radial Basis Function (RBF) Network.

This strategy uses a set of localized Gaussian radial basis functions centered at 
distinct points within the input domain, combined with a global linear trend.

Rationale:
1. Localized Approximation: Unlike global Chebyshev or Fourier bases, RBFs 
   allow the function to vary rapidly in specific regions without affecting 
   distant parts of the domain. This is crucial if the target function has 
   sharp features or non-smooth transitions.
2. Universal Approximation: A sufficiently wide set of Gaussian RBFs can 
   approximate any continuous function on a compact set.
3. Linear in Parameters: The model remains linear in the weights (params), 
   allowing BFGS to find the global optimum efficiently (equivalent to 
   solving a linear least-squares problem).
4. 10 Parameters Allocation:
   - 1 parameter for the global constant offset.
   - 1 parameter for the global linear trend.
   - 8 parameters for 8 Gaussian RBF centers.
   
   The centers are fixed at evenly spaced points across the domain [-3, 3].
   The widths of the Gaussians are fixed to provide sufficient overlap for 
   smooth interpolation while maintaining locality.

Basis functions:
- 1.0 (constant)
- x (linear)
- exp(-(x - c_i)^2 / (2 * sigma^2)) for i = 0..7, where c_i are fixed centers.

This covers a wide range of functions, including those with local variations.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the RBF expansion with a linear trend.
    
    Args:
        x: Input array of shape (N,)
        params: Coefficients array of shape (10,)
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    N = len(x)
    
    # Initialize basis matrix
    basis = np.zeros((N, 10))
    
    # --- Global Terms ---
    # Param 0: Constant
    basis[:, 0] = 1.0
    
    # Param 1: Linear
    basis[:, 1] = x
    
    # --- RBF Terms ---
    # 8 Gaussian centers evenly spaced in [-3, 3]
    # Centers: -3, -2.14, -1.28, -0.43, 0.43, 1.28, 2.14, 3
    # To avoid edge effects, we might shift them slightly inward or use a grid.
    # Let's use 8 points from -2.5 to 2.5 to ensure good coverage of the interior.
    centers = np.linspace(-2.5, 2.5, 8)
    
    # Width (sigma) of the Gaussians.
    # A sigma of 0.5 provides good locality while allowing overlap.
    sigma = 0.5
    
    for i in range(8):
        c = centers[i]
        # Gaussian: exp(-(x - c)^2 / (2 * sigma^2))
        basis[:, 2 + i] = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
    
    # Linear combination: y = basis @ params
    return basis @ params

# EVOLVE-BLOCK-END
