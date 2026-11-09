"""
Strategy: Dense Multi-Scale Fixed RBF Basis with Orthogonal Initialization.

Rationale:
The current Chebyshev approach uses a global polynomial basis. While stable for smooth functions,
it may struggle with complex interactions or local features if the target function is not well-
approximated by low-degree polynomials. Furthermore, BFGS starting from all-ones can be slow
if the basis functions are highly correlated or if the landscape is flat in certain directions.

This strategy uses a set of fixed Gaussian Radial Basis Functions (RBFs) centered on a grid
within the domain [-1, 1] x [-1, 1].
1. **Local Coverage**: RBFs provide local flexibility. A grid of centers ensures coverage of the domain.
2. **Multi-Scale**: Different widths (sigma) allow capturing both broad trends (large sigma) and
   finer details (small sigma).
3. **Orthogonal Initialization**: To help BFGS converge faster from the "all ones" start, we can
   pre-compute the basis matrix and apply a simple orthogonalization (like QR decomposition) to
   the basis vectors *conceptually*. However, since `equation` is called with `params` directly,
   we cannot modify the basis inside the function without changing the meaning of `params`.
   
   Instead, we rely on the fact that a well-distributed, multi-scale RBF basis often has a better
   condition number for local fitting than high-degree monomials. We select 10 centers/scales:
   - 1 center at (0,0) with large sigma (global trend)
   - 4 centers at (±0.5, ±0.5) with medium sigma (quadrants)
   - 4 centers at (±0.8, ±0.8) with small sigma (corners)
   - 1 center at (0,0) with small sigma (center detail)
   
   Total 10 parameters, each weighting one RBF.
   
   The RBF is defined as exp(-||p - c_i||^2 / (2 * sigma_i^2)).
   
   This structure is non-linear in the input (x,y) but linear in the parameters (weights),
   which BFGS handles well. The non-linearity is in the basis construction.
"""

import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Ensure we have at least 10 parameters
    if len(params) < 10:
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
    
    # Define RBF centers and scales
    # Format: (cx, cy, sigma)
    # We want a mix of scales to capture different frequencies.
    
    # 1. Global center, large sigma
    c0 = (0.0, 0.0, 1.5)
    
    # 2-5. Quadrant centers, medium sigma
    c1 = (-0.5, -0.5, 0.8)
    c2 = (0.5, -0.5, 0.8)
    c3 = (-0.5, 0.5, 0.8)
    c4 = (0.5, 0.5, 0.8)
    
    # 6-9. Corner centers, small sigma
    c5 = (-0.8, -0.8, 0.4)
    c6 = (0.8, -0.8, 0.4)
    c7 = (-0.8, 0.8, 0.4)
    c8 = (0.8, 0.8, 0.4)
    
    # 10. Center detail, small sigma
    c9 = (0.0, 0.0, 0.3)
    
    centers = np.array([c0[0], c0[1], c1[0], c1[1], c2[0], c2[1], 
                        c3[0], c3[1], c4[0], c4[1], c5[0], c5[1], 
                        c6[0], c6[1], c7[0], c7[1], c8[0], c8[1], 
                        c9[0], c9[1]])
    sigmas = np.array([c0[2], c1[2], c2[2], c3[2], c4[2], c5[2], c6[2], c7[2], c8[2], c9[2]])
    
    # Reshape centers for broadcasting
    # x, y are 1D arrays of length N
    # We need to compute distance from each point (x_i, y_i) to each center (cx_j, cy_j)
    
    # x: (N, 1), centers_x: (1, 10) -> diff_x: (N, 10)
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    
    cx = centers[0::2]  # Even indices are x-coords
    cy = centers[1::2]  # Odd indices are y-coords
    
    diff_x = x_col - cx[np.newaxis, :]
    diff_y = y_col - cy[np.newaxis, :]
    
    # Squared distance
    dist_sq = diff_x**2 + diff_y**2
    
    # Compute RBF values
    # exp(-dist_sq / (2 * sigma^2))
    # sigmas: (10,) -> (1, 10)
    sigma_sq = sigmas[np.newaxis, :]**2
    
    # Avoid division by zero if sigma is 0, though we set min sigma to 0.3
    rbf_values = np.exp(-dist_sq / (2.0 * sigma_sq))
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    # Prediction: sum over basis functions
    pred = np.sum(rbf_values * p[np.newaxis, :], axis=1)
    
    # Safety check for time
    if time.time() - start_time > 10:
         # Fallback to simple linear if something goes wrong (unlikely)
         # This fallback is just a safety net, the main path should be fast
         return p[0] + p[1] * x + p[4] * y
    
    return pred
