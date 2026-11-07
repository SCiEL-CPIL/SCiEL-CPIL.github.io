import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Low-degree Polynomial (global trend) + Gaussian RBFs (local features).
    
    This structure separates the approximation into:
    1. A cubic polynomial in t (mapped x), capturing smooth global behavior.
    2. Two Gaussian Radial Basis Functions, capturing local non-linearities.
    
    Parameters (10 total):
    params[0:4] : Polynomial coefficients c0, c1, c2, c3 for 1, t, t^2, t^3
    params[4:7] : RBF 1 parameters: amplitude a1, center c1, width raw w1
    params[7:10]: RBF 2 parameters: amplitude a2, center c2, width raw w2
    
    Transformations:
    - t = x / 3.0
    - RBF centers: c_i = tanh(raw_c_i) to keep in [-1, 1]
    - RBF widths: w_i = softplus(raw_w_i) + 0.1 to ensure positive, non-zero width
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map input to [-1, 1] for numerical stability
    t = x / 3.0
    
    # --- Polynomial Part ---
    # Basis: 1, t, t^2, t^3
    # We use raw params for coefficients.
    c0, c1, c2, c3 = params[0], params[1], params[2], params[3]
    y_poly = c0 + c1 * t + c2 * (t ** 2) + c3 * (t ** 3)
    
    # --- RBF Part ---
    # RBF 1
    a1 = params[4]
    c1_rbf = np.tanh(params[5])  # Center in [-1, 1]
    w1_raw = params[6]
    # Softplus: log(1 + exp(x))
    # Use stable softplus to avoid overflow
    w1 = np.log1p(np.exp(np.clip(w1_raw, -50, 50))) + 0.1
    
    # RBF 2
    a2 = params[7]
    c2_rbf = np.tanh(params[8])  # Center in [-1, 1]
    w2_raw = params[9]
    w2 = np.log1p(np.exp(np.clip(w2_raw, -50, 50))) + 0.1
    
    # Calculate RBF terms: exp(-0.5 * ((t - center) / width)^2)
    # Note: Using (t-center) since t is in [-1,1]
    diff1 = t - c1_rbf
    rbf1 = np.exp(-0.5 * (diff1 / w1) ** 2)
    
    diff2 = t - c2_rbf
    rbf2 = np.exp(-0.5 * (diff2 / w2) ** 2)
    
    y_rbf = a1 * rbf1 + a2 * rbf2
    
    result = y_poly + y_rbf
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
