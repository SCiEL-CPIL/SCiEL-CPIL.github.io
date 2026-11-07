import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    Architecture: Hybrid model combining a low-degree Legendre polynomial basis 
    (for global trend and numerical stability) with a Radial Basis Function (RBF) 
    component (for local flexibility).
    
    The RBF component uses 5 fixed, evenly spaced centers in [-3, 3] and a single 
    learnable width parameter. This avoids the instability of learnable frequencies 
    found in the previous design while retaining the ability to fit non-polynomial 
    features. The width is parameterized via a softplus-like transformation to 
    ensure positivity and smoothness for BFGS optimization.
    
    Parameters (10 total):
      params[0]: constant (P_0 coefficient)
      params[1]: linear (P_1 coefficient)
      params[2]: quadratic (P_2 coefficient)
      params[3]: cubic (P_3 coefficient)
      params[4]: raw width parameter for RBFs (transformed to positive width)
      params[5]: amplitude for RBF centered at -3.0
      params[6]: amplitude for RBF centered at -1.5
      params[7]: amplitude for RBF centered at 0.0
      params[8]: amplitude for RBF centered at 1.5
      params[9]: amplitude for RBF centered at 3.0
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Extract parameters
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    c3 = p[3]
    raw_width = p[4]
    amp_rbf_0 = p[5]
    amp_rbf_1 = p[6]
    amp_rbf_2 = p[7]
    amp_rbf_3 = p[8]
    amp_rbf_4 = p[9]
    
    # --- Polynomial Component (Legendre Basis) ---
    # Map x from [-3, 3] to t in [-1, 1] for Legendre polynomials
    t = x / 3.0
    
    # Legendre Polynomials P_n(t)
    P0 = np.ones_like(t)
    P1 = t
    P2 = 0.5 * (3.0 * t**2 - 1.0)
    P3 = 0.5 * (5.0 * t**3 - 3.0 * t)
    
    # Polynomial component
    y_poly = c0 * P0 + c1 * P1 + c2 * P2 + c3 * P3
    
    # --- RBF Component ---
    # Fixed centers evenly spaced in [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Transform raw_width to a positive width.
    # We want the width to be smooth and differentiable.
    # Using np.log1p(np.exp(abs(raw_width))) ensures width > 0 and is smooth.
    # However, we need to be careful with scaling. 
    # A typical Gaussian RBF is exp(-||x-c||^2 / (2*sigma^2)).
    # Let's define width = sigma^2.
    # To keep sigma in a reasonable range (e.g., 0.1 to 10), we can map raw_width.
    # Let's use a softplus-like mapping: width = np.log1p(np.exp(raw_width))
    # This maps (-inf, inf) to (0, inf).
    # If raw_width is 1 (initial), width = log(2) ~ 0.69.
    # If raw_width is 0, width = log(1) = 0? No, log1p(exp(0)) = log(2).
    # Let's use: width = 1.0 + np.log1p(np.exp(raw_width))
    # This ensures width > 1.0, which might be too wide.
    # Let's try: width = np.log1p(np.exp(raw_width)) * 2.0
    # Initial raw_width=1 -> width ~ 1.38.
    # This seems reasonable for a domain of size 6.
    
    width = 2.0 * np.log1p(np.exp(np.clip(raw_width, -20, 20)))
    
    # Compute RBF values
    # x is shape (N,), centers is shape (5,)
    # diff is shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian RBF: exp(-diff^2 / (2 * width))
    # Clip the exponent to avoid underflow/overflow issues, though exp(-large) is just 0.
    exponent = -0.5 * (diff ** 2) / width
    exponent = np.clip(exponent, -50.0, 0.0)
    
    rbf_values = np.exp(exponent)  # Shape (N, 5)
    
    # Combine RBFs with their amplitudes
    amps = np.array([amp_rbf_0, amp_rbf_1, amp_rbf_2, amp_rbf_3, amp_rbf_4])
    y_rbf = rbf_values @ amps
    
    # --- Combine Components ---
    result = y_poly + y_rbf
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
