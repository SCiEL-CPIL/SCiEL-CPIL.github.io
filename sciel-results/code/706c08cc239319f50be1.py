import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid physics + Gaussian RBF model for nonlinear oscillator.
    
    Structural change: replace global polynomial basis with local Gaussian RBFs
    in (t, x, v) space, combined with a few physics-motivated terms.
    
    Params (10 total):
      p0: bias
      p1: linear x coefficient
      p2: linear v coefficient  
      p3: sin(t) forcing coefficient
      p4-p9: weights for 6 RBFs with fixed centers spread in (t,x,v) space
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Physics-inspired terms (well-conditioned for BFGS from ones)
    result = p0 + p1 * x + p2 * v + p3 * np.sin(t)
    
    # RBF centers: spread across input ranges
    # t in [30, 50], x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    # 6 centers chosen to cover the space
    t_centers = np.array([35.0, 40.0, 45.0, 32.5, 42.5, 37.5])
    x_centers = np.array([-0.1, 0.0, 0.1, -0.05, 0.05, 0.0])
    v_centers = np.array([-0.2, 0.0, 0.2, 0.1, -0.1, 0.0])
    
    # Fixed Gaussian widths (chosen to cover ~40% of each input range)
    sigma_t = 5.0
    sigma_x = 0.08
    sigma_v = 0.15
    
    # RBF weights
    rbf_weights = np.array([p4, p5, p6, p7, p8, p9])
    
    # Compute RBF contributions efficiently
    for i in range(6):
        dt = (t - t_centers[i]) / sigma_t
        dx = (x - x_centers[i]) / sigma_x
        dv = (v - v_centers[i]) / sigma_v
        rbf_val = np.exp(-0.5 * (dt**2 + dx**2 + dv**2))
        result = result + rbf_weights[i] * rbf_val
    
    return result
