import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Cubic Polynomial + 3 Parametric Gaussian RBFs.
    
    Parameters (length 10):
    0-3: Coefficients for polynomial terms [1, x, x^2, x^3]
    4, 5: Center (c1) and Weight (w1) for RBF 1
    6, 7: Center (c2) and Weight (w2) for RBF 2
    8, 9: Center (c3) and Weight (w3) for RBF 3
    
    The RBFs use a fixed width sigma=1.0 to prevent ill-conditioning,
    but their centers are free to move, allowing the model to adapt
    to local features of the target function that a fixed basis would miss.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure exactly 10 params
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
        
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    c1 = params[4]
    w1 = params[5]
    c2 = params[6]
    w2 = params[7]
    c3 = params[8]
    w3 = params[9]
    
    # Polynomial part: captures global trend
    y = p0 + p1 * x + p2 * (x ** 2) + p3 * (x ** 3)
    
    # RBF part: captures local non-linearities
    # Fixed sigma = 1.0 for stability
    sigma_sq = 2.0  # 2 * sigma^2
    
    # RBF 1
    y += w1 * np.exp(-((x - c1) ** 2) / sigma_sq)
    
    # RBF 2
    y += w2 * np.exp(-((x - c2) ** 2) / sigma_sq)
    
    # RBF 3
    y += w3 * np.exp(-((x - c3) ** 2) / sigma_sq)
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
        
    return y
