"""
Strategy: Localized RBF Basis with Fixed Centers and Varied Bandwidths.

Rationale:
The previous approach used global basis functions (polynomials and low-frequency trigonometric terms).
While stable, these can struggle to capture sharp local features or complex non-polynomial shapes
without high-degree terms that cause ill-conditioning.

This new approach uses a set of Gaussian Radial Basis Functions (RBFs) centered at fixed points
across the domain [0.1, 6.0].
1. **Locality**: Each parameter primarily influences the prediction near its center. This decouples
   the optimization of local features, making the problem easier for BFGS to navigate.
2. **Flexibility**: By varying the bandwidths (widths) of the Gaussians, we can capture both
   broad trends (wide bandwidths) and sharp details (narrow bandwidths).
3. **Linearity**: The model remains linear in the parameters, ensuring convexity and global
   convergence of the least-squares fit.

Design:
- 10 parameters total.
- 1 Constant term (to handle global offset).
- 9 RBF terms:
  - Centers are spaced roughly evenly across [0.5, 5.5] to cover the domain well.
  - Bandwidths are varied: a mix of wide (for trend) and narrow (for detail) Gaussians.
  - Specifically, we use a geometric progression of bandwidths or a specific set to ensure
    coverage of different scales.

The equation is:
y = p[0] + sum_{i=1}^{9} p[i] * exp(-0.5 * ((x - c_i) / w_i)^2)

This structure is robust to noise and allows BFGS to quickly zero out irrelevant bumps
and adjust the relevant ones.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Precompute centers and bandwidths to avoid overhead in the equation call.
# Domain is [0.1, 6.0].
# We want centers to cover the domain.
# Let's place 9 centers.
# c_i = 0.5 + i * (5.0 / 8) for i in 0..8? 
# Range 0.5 to 5.5. Step 0.625.
# Let's refine centers to be slightly more dense in the middle or just uniform.
# Uniform is safer.
centers = np.linspace(0.5, 5.5, 9)

# Bandwidths: We need a mix.
# If all are same, it's like a fixed-width RBF.
# Let's use a mix:
# 3 wide (w=2.0), 3 medium (w=1.0), 3 narrow (w=0.5)
# Or a geometric spread.
# Let's try a specific pattern to ensure scale invariance.
# w = [2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
# This allows capturing broad trends and sharp peaks.
bandwidths = np.array([2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2])

# Precompute the inverse squared bandwidths for efficiency: -0.5 / w^2
inv_w2 = -0.5 / (bandwidths * bandwidths)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a constant term and 9 Gaussian RBF terms.
    params: array of shape (10,)
      params[0]: constant offset
      params[1:10]: weights for the 9 RBFs
    x: Input array of shape (N,)
    Returns: array of shape (N,)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Clip x to domain to prevent extrapolation artifacts, though Gaussians decay naturally.
    # Clipping ensures stability if x is slightly out of bounds due to numerical noise.
    x = np.clip(x, 0.1, 6.0)
    
    # Start with the constant term
    pred = np.full_like(x, params[0], dtype=np.float64)
    
    # Add RBF contributions
    # Vectorized computation:
    # diff: shape (N, 9)
    # We can compute all 9 RBFs at once.
    
    # Create a 2D array of differences: x[:, None] - centers[None, :]
    diff = x[:, None] - centers[None, :]
    
    # Compute squared differences
    diff_sq = diff * diff
    
    # Compute exponents: diff_sq * inv_w2
    # inv_w2 is shape (9,), diff_sq is (N, 9). Broadcasting works.
    exponents = diff_sq * inv_w2[None, :]
    
    # Compute Gaussians: exp(exponents)
    # Clip exponents to avoid underflow warnings (though exp(-large) is just 0)
    # and overflow (exp(+large) is inf, but here exponents are always <= 0 because diff_sq >= 0 and inv_w2 < 0)
    # So exponents are in [-inf, 0]. exp is safe.
    rbf_values = np.exp(exponents)
    
    # Weighted sum: rbf_values @ params[1:]
    # params[1:] is shape (9,)
    weights = params[1:]
    
    # Add to prediction
    pred += rbf_values.dot(weights)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred

# EVOLVE-BLOCK-END
