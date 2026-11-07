"""
Strategy: Gaussian RBF Network
Instead of a global polynomial expansion (which forces the entire function to be smooth and low-order globally), 
we use a set of Gaussian Radial Basis Functions. 
This allows the model to fit local features of the nonlinear oscillator (like the limit cycle boundary) 
more effectively. The centers are fixed on a grid in the normalized (x, v) space, 
and the 10 parameters are the weights of these centers plus a bias.
"""
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Gaussian RBF basis.
    
    Inputs:
    x: displacement
    v: velocity
    params: array of 10 weights [w0, ..., w8, bias]
    
    The basis functions are Gaussian kernels centered on a 3x3 grid 
    in the normalized (x, v) domain.
    """
    # Normalization to standard range roughly [-1, 1] or similar scale
    # x range: [-0.5728, 0.5275] -> center ~ -0.02, scale ~ 0.55
    # v range: [-0.1754, 0.2821] -> center ~ 0.05, scale ~ 0.25
    x_c = -0.02
    v_c = 0.05
    x_s = 0.55
    v_s = 0.25
    
    x_n = (x - x_c) / x_s
    v_n = (v - v_c) / v_s
    
    # Define 9 fixed centers on a 3x3 grid in normalized space
    # Centers: -1, 0, 1
    centers_x = np.array([-1.0, 0.0, 1.0])
    centers_v = np.array([-1.0, 0.0, 1.0])
    
    # Create meshgrid for centers
    cx, cv = np.meshgrid(centers_x, centers_v, indexing='ij')
    cx = cx.flatten()
    cv = cv.flatten()
    
    # Compute distances from each input point to each center
    # x_n shape: (N, 1) or (N,)
    # We need to handle both 1D and 2D inputs, but usually 1D (N,)
    # Reshape to (N, 1) for broadcasting
    x_col = x_n.reshape(-1, 1)
    v_col = v_n.reshape(-1, 1)
    
    # Distance squared: (N, 1) - (1, 9) -> (N, 9)
    dx = x_col - cx.reshape(1, -1)
    dv = v_col - cv.reshape(1, -1)
    
    dist_sq = dx**2 + dv**2
    
    # Gaussian width. 
    # If width is too small, the function is too spiky. If too large, it's too global.
    # A width of ~0.8-1.0 in normalized space is reasonable for 3x3 grid.
    sigma_sq = 1.0 
    # Using exp(-dist_sq / (2 * sigma^2))
    # Let's use a standard Gaussian: exp(-0.5 * dist_sq)
    basis = np.exp(-0.5 * dist_sq)
    
    # basis shape: (N, 9)
    
    # params: (10,)
    # First 9 are weights, last is bias
    weights = params[:9]
    bias = params[9]
    
    # Weighted sum: (N, 9) * (9,) -> (N,)
    out = basis @ weights + bias
    
    return out
