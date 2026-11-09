import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a hidden function using a Radial Basis Function (RBF) model.
    
    Strategy:
    - Instead of a global polynomial, use 10 fixed Gaussian basis functions (bumps).
    - The centers of these bumps are distributed across the input domain [0.5, 2.0]^3.
    - The 10 parameters are the weights (amplitudes) for each bump.
    - This allows the model to capture local variations in the target function more effectively
      than a low-order global polynomial, while keeping the parameter count at 10.
    - The model is linear in the parameters, which often aids optimization stability.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Extract parameters (first 10)
    p = params[:10]
    
    # Define centers for the 10 RBFs.
    # We want to cover the domain [0.5, 2.0] in 3D.
    # A simple strategy is to use a grid or a set of well-spaced points.
    # Let's use a combination of corner-ish and center points to ensure coverage.
    # Domain size is 1.5 in each dimension.
    # Centers:
    # 1. (0.5, 0.5, 0.5) - Corner
    # 2. (2.0, 2.0, 2.0) - Opposite Corner
    # 3. (1.25, 1.25, 1.25) - Center
    # 4. (0.5, 2.0, 1.25) - Edge/Mid
    # 5. (2.0, 0.5, 1.25) - Edge/Mid
    # 6. (1.25, 0.5, 2.0) - Edge/Mid
    # 7. (1.25, 2.0, 0.5) - Edge/Mid
    # 8. (0.75, 0.75, 2.0) - Near corner
    # 9. (1.75, 1.75, 0.5) - Near opposite corner
    # 10. (0.75, 1.75, 1.25) - Mixed
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [0.5, 2.0, 1.25],
        [2.0, 0.5, 1.25],
        [1.25, 0.5, 2.0],
        [1.25, 2.0, 0.5],
        [0.75, 0.75, 2.0],
        [1.75, 1.75, 0.5],
        [0.75, 1.75, 1.25]
    ])
    
    # Stack input vectors into a matrix of shape (N, 3)
    # N is the number of data points
    inputs = np.column_stack((x, y, z))
    
    # Calculate squared distances from each point to each center
    # inputs: (N, 3)
    # centers: (10, 3)
    # diff: (N, 10, 3)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    # Squared Euclidean distances
    sq_dist = np.sum(diff ** 2, axis=2)  # Shape: (N, 10)
    
    # Define the width (sigma) of the Gaussians.
    # The domain is [0.5, 2.0], range 1.5.
    # A sigma of 0.4 or 0.5 allows some overlap but maintains locality.
    # If sigma is too small, the model becomes too rigid (local).
    # If sigma is too large, it becomes global (like a polynomial).
    # Let's try sigma = 0.5.
    sigma = 0.5
    sigma_sq = sigma ** 2
    
    # Calculate Gaussian values: exp(-sq_dist / (2 * sigma_sq))
    # To ensure numerical stability and avoid underflow/overflow issues in BFGS,
    # we compute the basis values.
    basis = np.exp(-sq_dist / (2.0 * sigma_sq))  # Shape: (N, 10)
    
    # The prediction is the dot product of basis values and parameters
    # basis: (N, 10), p: (10,)
    predictions = basis @ p
    
    # Ensure output is 1D array as required
    return predictions
