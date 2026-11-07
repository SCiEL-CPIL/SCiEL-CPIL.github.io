"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF + Polynomial basis.
    Uses 5 Gaussian RBFs with fixed centers and widths to capture local features,
    and 5 polynomial terms to capture global trends.
    This structure is fundamentally different from pure polynomial or trigonometric bases,
    allowing the model to fit localized "bumps" or non-smooth transitions that polynomials
    cannot handle efficiently with few parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # 1. RBF Component (5 parameters)
    # Fixed centers on a grid
    centers = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5]
    ])
    
    # Fixed width (sigma = 0.5, so 2*sigma^2 = 0.5, inverse is 2.0)
    # Using a moderate width to avoid being too local or too global
    sigma_sq_inv = 2.0 
    
    rbf_pred = np.zeros_like(x)
    for i in range(5):
        cx, cy = centers[i]
        dist_sq = (x - cx)**2 + (y - cy)**2
        # Gaussian kernel
        kernel = np.exp(-sigma_sq_inv * dist_sq)
        rbf_pred += params[i] * kernel
        
    # 2. Polynomial Component (5 parameters)
    # Global trend
    p_const = params[5]
    p_x = params[6]
    p_y = params[7]
    p_xy = params[8]
    p_quad = params[9]
    
    poly_pred = (p_const + 
                 p_x * x + 
                 p_y * y + 
                 p_xy * x * y + 
                 p_quad * (x**2 + y**2))
                 
    return rbf_pred + poly_pred
# EVOLVE-BLOCK-END
