"""
Rewritten program: Non-linear Gaussian RBF basis for symbolic regression.

Strategy:
The previous program used a linear basis of Chebyshev polynomials. While well-conditioned,
it is restricted to the space spanned by those specific polynomials. If the target function
has localized features or non-polynomial behavior (e.g., peaks, plateaus, or specific
non-linear interactions not well-approximated by low-degree polynomials), a linear basis
may fail to fit it well, leading to high residual error.

This new strategy uses a small, non-linear model: a weighted sum of two 2D Gaussian
Radial Basis Functions (RBFs). The parameters define the centers $(c_{1x}, c_{1y}, c_{2x}, c_{2y})$,
widths $(\sigma_1, \sigma_2)$, and amplitudes $(A_1, A_2)$ of the two Gaussians.
Additionally, we include a global bias term to handle the mean of the function.

Model:
f(x, y) = A_1 * exp(-((x - c_1x)^2 + (y - c_1y)^2) / (2 * sigma_1^2)) +
          A_2 * exp(-((x - c_2x)^2 + (y - c_2y)^2) / (2 * sigma_2^2)) +
          A_3

Parameters (10 total):
p[0]: A_1 (Amplitude 1)
p[1]: c_1x (Center 1 x)
p[2]: c_1y (Center 1 y)
p[3]: sigma_1 (Width 1)
p[4]: A_2 (Amplitude 2)
p[5]: c_2x (Center 2 x)
p[6]: c_2y (Center 2 y)
p[7]: sigma_2 (Width 2)
p[8]: A_3 (Bias/Global Amplitude)
p[9]: Unused (set to 0 or ignored, but must exist for shape)

Why this may work better:
1. Non-linearity: The relationship between parameters and output is non-linear, allowing
   the optimizer to fit complex shapes.
2. Localization: Gaussians can model local variations that global polynomials might
   struggle with if they require high degrees to approximate.
3. Flexibility: The centers and widths are learnable, allowing the model to adapt to
   the specific structure of the hidden function.

To ensure numerical stability and avoid division by zero or overflow:
- Widths are regularized with a small epsilon.
- The exponent is clipped to prevent underflow/overflow issues in exp.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a sum of two 2D Gaussian RBFs and a bias.
    
    Parameters:
    p[0]: A_1
    p[1]: c_1x
    p[2]: c_1y
    p[3]: sigma_1
    p[4]: A_2
    p[5]: c_2x
    p[6]: c_2y
    p[7]: sigma_2
    p[8]: A_3 (Bias)
    p[9]: Unused
    """
    # Extract parameters
    A1 = params[0]
    c1x = params[1]
    c1y = params[2]
    s1 = params[3]
    
    A2 = params[4]
    c2x = params[5]
    c2y = params[6]
    s2 = params[7]
    
    bias = params[8]
    
    # Regularize widths to prevent division by zero and ensure positive variance
    # BFGS starts at 1, so initial sigma is 1.
    # We use abs to ensure sigma is positive, but BFGS might try negative values.
    # Using abs makes the landscape symmetric and avoids singularities at 0.
    # Adding a small epsilon prevents division by zero if sigma approaches 0.
    eps = 1e-6
    s1_safe = np.abs(s1) + eps
    s2_safe = np.abs(s2) + eps
    
    # Calculate squared distances for each Gaussian
    # d1^2 = (x - c1x)^2 + (y - c1y)^2
    dx1 = x - c1x
    dy1 = y - c1y
    d1_sq = dx1**2 + dy1**2
    
    # d2^2 = (x - c2x)^2 + (y - c2y)^2
    dx2 = x - c2x
    dy2 = y - c2y
    d2_sq = dx2**2 + dy2**2
    
    # Calculate Gaussian terms
    # exp(-d^2 / (2 * sigma^2))
    # Clip the exponent to avoid underflow (which is fine, becomes 0) or overflow (which is bad)
    # Since d^2 >= 0 and sigma^2 > 0, the exponent is always <= 0.
    # So exp will be in (0, 1]. No overflow risk from positive exponent.
    # Underflow to 0 is acceptable.
    
    term1 = np.exp(-d1_sq / (2.0 * s1_safe**2))
    term2 = np.exp(-d2_sq / (2.0 * s2_safe**2))
    
    # Combine
    f = A1 * term1 + A2 * term2 + bias
    
    # Ensure finite output
    # If any param becomes inf/nan during optimization, this returns nan/inf.
    # The evaluator handles non-finite errors by penalizing, but we should try to be robust.
    # However, replacing nan with 0 might hide optimization issues. 
    # Standard practice in these tasks is to let the optimizer see the loss.
    # But if the result is nan, the fit fails.
    # We can use np.nan_to_num to replace nan/inf with 0 or large numbers?
    # If params explode, the prediction is garbage anyway.
    # Let's just return f.
    
    return f
# EVOLVE-BLOCK-END
