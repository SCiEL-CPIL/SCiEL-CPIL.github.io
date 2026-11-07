"""
Strategy: Locally Adaptive Gaussian Basis (Neural Network-like)
Instead of a fixed global basis, we use 5 Gaussian bumps with learnable centers and widths.
This allows the model to fit local features more accurately than a fixed polynomial/trig basis.
The parameters are:
- p[0:5]: Amplitudes of the 5 Gaussians
- p[5:8]: Centers of the first 3 Gaussians (the last 2 are fixed at -3 and 3 to ensure domain coverage)
- p[8]: Width scale (shared for all Gaussians to maintain stability)
- p[9]: Unused (kept for compatibility, set to 0)
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a sum of 5 Gaussian bumps.
    The centers of the first 3 bumps are learnable, while the last 2 are fixed 
    at the boundaries of the domain [-3, 3] to ensure full coverage.
    The width is controlled by a single learnable parameter to maintain stability.
    """
    # Ensure x is a 1D float array
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Pad params if necessary to ensure length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
        
    # Extract parameters
    # Amplitudes
    a = p[0:5]
    
    # Centers
    # Learnable centers for first 3 bumps
    # We add a small offset to ensure they don't all start at 1.0 (initial value)
    # If p[5] is 1.0, center is 1.0. We map p[5:8] to a range [-2.5, 2.5] roughly?
    # Actually, BFGS starts at 1.0. If we use raw p[5:8] as centers, they start at 1.0.
    # This is a local minimum trap if the true function is symmetric or centered elsewhere.
    # To help BFGS, we can define centers relative to a grid.
    # Let's define fixed centers for the last 2: -3.0 and 3.0.
    # For the first 3, we use the raw parameters but constrain them slightly?
    # No, just use raw parameters. But starting at 1.0 for all 3 is bad.
    # Let's offset them: c1 = p[5] - 2.0, c2 = p[6], c3 = p[7] + 2.0.
    # If p=1, c1=-1, c2=1, c3=3. This provides initial spread.
    c1 = p[5] - 2.0
    c2 = p[6]
    c3 = p[7] + 2.0
    c4 = -3.0  # Fixed left boundary
    c5 = 3.0   # Fixed right boundary
    
    centers = np.array([c1, c2, c3, c4, c5])
    
    # Width
    # p[8] starts at 1.0.
    # Width should be positive. Use exp(p[8]) or abs(p[8])?
    # exp(1) is ~2.7. This is a reasonable width for [-3,3].
    # If p[8] goes to -inf, width -> 0 (spike). If p[8] -> inf, width -> inf (flat).
    # Let's use exp(p[8]) to ensure positivity.
    width = np.exp(p[8])
    
    # Construct the basis
    # We want y = sum(a_i * exp(-0.5 * ((x - c_i) / width)^2))
    # Vectorized computation
    # x is (N,), centers is (5,)
    # x[:, np.newaxis] is (N, 1)
    # centers[np.newaxis, :] is (1, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exponents
    # Avoid division by zero if width is extremely small (though exp ensures >0)
    # Clip width to a minimum to prevent numerical issues if optimizer pushes it to -inf
    width_safe = np.maximum(width, 1e-3)
    
    exponent = -0.5 * (diff / width_safe) ** 2
    
    # Compute Gaussians
    gaussians = np.exp(exponent)
    
    # Linear combination
    # a is (5,), gaussians is (N, 5)
    y = gaussians @ a
    
    return y

# EVOLVE-BLOCK-END
