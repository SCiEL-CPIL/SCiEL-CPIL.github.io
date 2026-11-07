import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Gaussian RBF basis.
    
    The model uses 10 parameters:
      p0: global bias
      p1-p9: weights for 9 Gaussian RBF centers in the (t, x, v) space
    
    RBF centers are placed at distinct locations within the input ranges
    to provide smooth, localized nonlinear basis functions.
    """
    # Input ranges from the task:
    # t: [30, 50], x: [-0.3391, 0.3624], v: [-0.2056, 0.1749]
    
    # Define 9 RBF centers in the 3D input space (t, x, v)
    # t centers spread across [30, 50]
    # x centers spread across [-0.3391, 0.3624]
    # v centers spread across [-0.2056, 0.1749]
    
    t_centers = np.array([32.5, 37.5, 42.5, 47.5, 35.0, 40.0, 45.0, 30.0, 50.0])
    x_centers = np.array([-0.25, 0.0, 0.25, -0.30, 0.30, -0.15, 0.15, -0.20, 0.20])
    v_centers = np.array([-0.15, 0.0, 0.15, -0.20, 0.10, -0.10, 0.05, -0.18, 0.12])
    
    # Widths scaled to domain size
    # t domain size: 20, x domain size: ~0.7, v domain size: ~0.38
    # Use moderate widths to allow overlap between RBFs
    t_width = 5.0   # ~25% of t range
    x_width = 0.25  # ~35% of x range
    v_width = 0.12  # ~30% of v range
    
    p0 = params[0]
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = params[1:]
    
    # Compute squared distances in normalized space
    dt = (t - t_centers) / t_width
    dx = (x - x_centers) / x_width
    dv = (v - v_centers) / v_width
    
    # Gaussian RBF values: exp(-0.5 * (dt^2 + dx^2 + dv^2))
    rbf1 = np.exp(-0.5 * (dt**2 + dx**2 + dv**2))
    
    # Use different center indices for each RBF to create distinct basis functions
    # RBF 1: centers[0], RBF 2: centers[1], etc.
    # But we need 9 distinct RBFs with 9 weight parameters
    
    # Recompute with distinct centers for each RBF
    dt1 = (t - t_centers[0]) / t_width
    dx1 = (x - x_centers[0]) / x_width
    dv1 = (v - v_centers[0]) / v_width
    rbf1 = np.exp(-0.5 * (dt1**2 + dx1**2 + dv1**2))
    
    dt2 = (t - t_centers[1]) / t_width
    dx2 = (x - x_centers[1]) / x_width
    dv2 = (v - v_centers[1]) / v_width
    rbf2 = np.exp(-0.5 * (dt2**2 + dx2**2 + dv2**2))
    
    dt3 = (t - t_centers[2]) / t_width
    dx3 = (x - x_centers[2]) / x_width
    dv3 = (v - v_centers[2]) / v_width
    rbf3 = np.exp(-0.5 * (dt3**2 + dx3**2 + dv3**2))
    
    dt4 = (t - t_centers[3]) / t_width
    dx4 = (x - x_centers[3]) / x_width
    dv4 = (v - v_centers[3]) / v_width
    rbf4 = np.exp(-0.5 * (dt4**2 + dx4**2 + dv4**2))
    
    dt5 = (t - t_centers[4]) / t_width
    dx5 = (x - x_centers[4]) / x_width
    dv5 = (v - v_centers[4]) / v_width
    rbf5 = np.exp(-0.5 * (dt5**2 + dx5**2 + dv5**2))
    
    dt6 = (t - t_centers[5]) / t_width
    dx6 = (x - x_centers[5]) / x_width
    dv6 = (v - v_centers[5]) / v_width
    rbf6 = np.exp(-0.5 * (dt6**2 + dx6**2 + dv6**2))
    
    dt7 = (t - t_centers[6]) / t_width
    dx7 = (x - x_centers[6]) / x_width
    dv7 = (v - v_centers[6]) / v_width
    rbf7 = np.exp(-0.5 * (dt7**2 + dx7**2 + dv7**2))
    
    dt8 = (t - t_centers[7]) / t_width
    dx8 = (x - x_centers[7]) / x_width
    dv8 = (v - v_centers[7]) / v_width
    rbf8 = np.exp(-0.5 * (dt8**2 + dx8**2 + dv8**2))
    
    dt9 = (t - t_centers[8]) / t_width
    dx9 = (x - x_centers[8]) / x_width
    dv9 = (v - v_centers[8]) / v_width
    rbf9 = np.exp(-0.5 * (dt9**2 + dx9**2 + dv9**2))
    
    result = p0 + p1*rbf1 + p2*rbf2 + p3*rbf3 + p4*rbf4 + p5*rbf5 + p6*rbf6 + p7*rbf7 + p8*rbf8 + p9*rbf9
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
    
    return result
