import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-Center Gaussian RBF Model.
    
    Strategy:
    - Use 9 Gaussian Radial Basis Functions centered on a 3x3 grid in [-1, 1]^2.
    - Add 1 constant term.
    - Total 10 parameters (9 weights + 1 bias).
    - The model is linear in the parameters, ensuring BFGS finds the global minimum.
    - This captures local non-linear features better than global polynomials, 
      while avoiding the non-convex optimization issues of variable-center RBFs.
    """
    # Fixed centers for 3x3 grid
    cx = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    cy = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    
    # Fixed width (sigma)
    # Chosen to balance locality and coverage. 
    # 0.5 allows significant overlap, ensuring smoothness.
    sigma = 0.5
    
    # Ensure x and y are 1D
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Compute squared distances to each center
    # diff_x shape: (N, 9)
    diff_x = x[:, np.newaxis] - cx[np.newaxis, :]
    diff_y = y[:, np.newaxis] - cy[np.newaxis, :]
    
    dist_sq = diff_x**2 + diff_y**2
    
    # Compute RBF values
    rbf_values = np.exp(-dist_sq / (2 * sigma**2))
    
    # Extract weights and constant
    w = params[:9]
    c = params[9]
    
    # Linear combination
    pred = rbf_values @ w + c
    
    return pred
