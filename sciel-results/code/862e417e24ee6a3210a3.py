"""
Strategy: Locally Weighted Kernel Regression (RBF) with Temperature-Dependent Centers.

Instead of using global basis functions (polynomials, tanh) which assume a specific
global shape (e.g., convex/concave, symmetric), we use a sum of Gaussian Radial Basis
Functions (RBFs). This is a universal approximator for smooth functions.

Key Structural Differences from Current Program:
1. Local Adaptation: The model can fit sharp features (like yield points) or flat
   regions (elastic/plastic plateaus) by placing kernels where needed, rather than
   relying on global terms like eps**2 or tanh(eps) which have fixed curvature profiles.
2. Temperature Coupling: The centers of the kernels shift with temperature,
   allowing the "yield point" or "knee" of the curve to move naturally with T,
   which is physically accurate for metals (yield strength drops with T).
3. Parameterization: 
   - 5 Kernels.
   - 2 params per kernel: Amplitude and Center (in strain space).
   - The centers are linearly dependent on Temperature: center_i = c_i + d_i * T_norm.
   - We use 5 amplitudes (p[0:5]) and 5 center-offsets (p[5:10]).
   - Wait, we need 10 params total. 
   - Let's use:
     - p[0:5]: Amplitudes of 5 kernels.
     - p[5:10]: Base centers of 5 kernels.
   - The width of the kernels is fixed to a value that covers the strain range well 
     (e.g., sigma = 0.25) to avoid overfitting and ensure smoothness. 
   - Temperature dependence: We can't have a separate param for T-shift for each 
     kernel if we only have 10 params. 
     
   Revised Parameterization for 10 Params:
   - p[0:5]: Amplitudes (A_i)
   - p[5:8]: Base Centers (C_i) for 3 main kernels (Elastic, Yield, Hardening)
   - p[8:10]: Temperature sensitivity coefficients (S_i) that shift the centers.
   
   Actually, simpler and more robust:
   - Use 4 Kernels.
   - p[0:4]: Amplitudes
   - p[4:8]: Centers
   - p[8:10]: Global temperature scaling and offset? 
   
   Let's try a different structural approach: **Piecewise Linear Spline with Smooth Joints via RBF**.
   
   Better Idea: **Anisotropic RBF in (Strain, Temp) space**.
   But we only have 10 params.
   
   Let's stick to the most robust "structural" change: **Gaussian Process-like Likelihood with Fixed Hyperparameters but Learnable Mean Function via RBF**.
   
   Let's use 5 Gaussian Kernels.
   - p[0:5]: Weights (Amplitudes)
   - p[5:10]: Centers (Mean strain locations)
   
   Where is Temperature?
   The current program handles T via linear scaling of coefficients.
   If we ignore T in the centers, we might miss the shift.
   
   Alternative: **Separable Model with RBF in Strain and Linear in Temp**.
   sigma(eps, T) = sum_i A_i * exp(-(eps - c_i)^2 / (2*sigma^2)) * (1 + B_i * T_norm)
   
   This requires 10 params:
   - 5 Amplitudes A_i
   - 5 Centers c_i
   - But where are the B_i? We don't have enough params.
   
   Let's use:
   - 4 Kernels.
   - p[0:4]: Amplitudes
   - p[4:8]: Centers
   - p[8]: Temperature scaling factor for the whole curve (or specific shift)
   - p[9]: Global offset
   
   This is still somewhat similar to the current "sum of basis functions" but the basis is local (Gaussian) instead of global (Polynomial/Tanh). The key difference is locality.
   
   To make it materially different in *how* it fits:
   The current program uses BFGS on the raw parameters.
   The new program will use **Log-space parameterization for amplitudes** to ensure positivity and stability, and **Tanh-space for centers** to keep them within the training range. This changes the optimization landscape significantly.
   
   Let's refine the RBF model:
   sigma = sum_{i=0}^{3} A_i * exp(-(eps - c_i)^2 / (2 * w^2)) * (1 + T_shift * T_norm) + offset
   
   Params:
   p[0:4] -> log(A_i) + 5 (to get positive A_i around 1)
   p[4:8] -> tanh(c_i) mapped to [-0.01, 1.0]
   p[8] -> T_shift (linear effect of T on the whole curve's magnitude or shift)
   p[9] -> offset
   
   Width w is fixed to 0.2 to ensure smoothness and avoid overfitting 2161 points with 4 kernels.
"""

import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Inputs:
    strain: array of strain values
    temp: array of temperature values
    params: array of 10 parameters
    
    Returns:
    array of stress values
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Normalize temperature to [0, 1]
    T_norm = (T - 0.06667) / (1.0 - 0.06667)
    T_norm = np.clip(T_norm, 0.0, 1.0)
    
    p = params
    
    # 1. Extract and transform parameters for stability
    # Amplitudes: Use exp to ensure positive weights, centered around 1.
    # p[0:4] are the log-amplitudes.
    A = np.exp(p[0:4])
    
    # Centers: Use tanh to map to a reasonable range.
    # Training strain range is [-0.001911, 0.9636].
    # We want centers to be able to cover this range.
    # Map tanh(p) from [-1, 1] to [-0.1, 1.1]
    c = 0.5 * (tanh(p[4:8]) + 1.0) * 1.2 - 0.1
    
    # Temperature effect:
    # p[8] controls how much the curve shifts or scales with temperature.
    # Let's model it as a multiplicative factor on the RBF sum.
    T_factor = 1.0 + p[8] * T_norm
    
    # p[9] is the global offset.
    offset = p[9]
    
    # 2. Calculate RBF sum
    # Fixed width for stability and smoothness.
    # A width of 0.2 allows kernels to cover ~1 standard deviation of the strain range.
    width = 0.2
    width_sq_2 = 2.0 * width * width
    
    # Vectorized calculation
    # eps shape: (N,)
    # c shape: (4,)
    # We need to compute exp(-(eps - c_i)^2 / width_sq_2) for each i
    
    # Expand dims for broadcasting: eps[:, None] - c[None, :]
    diff = eps[:, None] - c[None, :]
    
    # Compute Gaussians
    gaussians = np.exp(-diff * diff / width_sq_2)
    
    # Weighted sum
    # A shape: (4,) -> broadcast to (N, 4)
    sigma_rbf = np.sum(gaussians * A[None, :], axis=1)
    
    # Apply temperature factor and offset
    sigma = sigma_rbf * T_factor + offset
    
    # Ensure finite output
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
