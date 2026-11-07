# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10 localized Gaussian Radial Basis Functions (RBFs).
    
    params layout (length 10):
    - We use 5 RBFs to keep the parameter space manageable and reduce overfitting risk 
      with only 10 parameters, OR we can use 10 RBFs where each parameter controls 
      the amplitude of a fixed center/width pair.
      
    The directive suggests: "10 localized Radial Basis Functions (RBFs) centered at distinct points 
    in [-3, 3] with varying Gaussian widths, allowing the 10 parameters to independently adjust 
    local amplitude and shape".
    
    However, we only have 10 parameters total. 
    If we use 10 RBFs, we can only have 1 parameter per RBF (amplitude), with fixed centers and widths.
    Alternatively, we can use fewer RBFs with more parameters each (e.g., 5 RBFs with amplitude and width, 
    or amplitude and center).
    
    Let's try 5 RBFs. Each RBF gets 2 parameters: amplitude and width. 
    Centers will be fixed at evenly spaced points in [-3, 3]: -3, -1.5, 0, 1.5, 3.
    This gives 5 * 2 = 10 parameters.
    
    Or, let's try 10 RBFs with fixed centers and fixed widths, and 10 amplitude parameters.
    This is simpler and often effective. The directive says "varying Gaussian widths", but with 
    only 10 params, we can't vary both width and amplitude for 10 RBFs. 
    We can fix the widths to be reasonable (e.g., covering the domain) and let the amplitudes vary.
    Or we can fix the centers and let the widths vary? 
    Let's stick to the most robust interpretation: 
    Use 10 RBFs. 
    Centers: fixed, evenly spaced in [-3, 3].
    Widths: fixed, chosen to provide good coverage (e.g., sigma = 1.0 or based on spacing).
    Parameters: 10 amplitudes.
    
    Wait, the directive says "allowing the 10 parameters to independently adjust local amplitude and shape". 
    This implies the parameters control both. But 10 params for 10 RBFs means 1 param per RBF. 
    So "shape" might be fixed, or "amplitude and shape" is a loose description of the flexibility provided by the combination.
    
    Let's try a hybrid: 5 RBFs.
    Centers: fixed at -2.5, -1.25, 0, 1.25, 2.5.
    For each RBF, we have 2 parameters: amplitude (a) and inverse width (b).
    RBF_i(x) = a_i * exp(-b_i * (x - c_i)^2)
    This uses 10 parameters total.
    This allows the optimizer to adjust both the height and the width of each local bump.
    """
    # Fixed centers
    centers = np.array([-2.5, -1.25, 0.0, 1.25, 2.5])
    
    # Extract parameters
    # p0, p1 for RBF 0
    # p2, p3 for RBF 1
    # p4, p5 for RBF 2
    # p6, p7 for RBF 3
    # p8, p9 for RBF 4
    
    amplitudes = params[0::2]  # p0, p2, p4, p6, p8
    inv_widths = params[1::2]  # p1, p3, p5, p7, p9
    
    # Ensure widths are positive to avoid instability? 
    # The optimizer might try negative widths. exp(-b * x^2) with b<0 diverges.
    # We should use abs(inv_widths) or a softplus-like transform.
    # Using abs is simple and differentiable almost everywhere.
    # Or better: use a parameterization that guarantees positivity.
    # Let b = exp(inv_width_param). But then the optimizer works in log-space.
    # Or b = 1 + abs(inv_width_param).
    # Let's use abs() to keep it simple, or just let the optimizer figure it out.
    # If b is negative, the Gaussian becomes a "Gaussian" that grows at infinity, which is bad.
    # Let's force b to be positive: b = abs(inv_widths)
    
    b = np.abs(inv_widths)
    
    # Compute RBFs
    # x shape: (N,)
    # centers shape: (5,)
    # We need to compute sum_{i=0}^{4} a_i * exp(-b_i * (x - c_i)^2)
    
    # Vectorized computation
    # x: (N, 1)
    # centers: (1, 5)
    # diff: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # b: (5,) -> (1, 5)
    # a: (5,) -> (1, 5)
    rbf_values = amplitudes[np.newaxis, :] * np.exp(-b[np.newaxis, :] * diff**2)
    
    # Sum over RBFs
    y_pred = np.sum(rbf_values, axis=1)
    
    return y_pred
# EVOLVE-BLOCK-END
