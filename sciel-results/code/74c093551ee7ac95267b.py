import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Radial Basis Function (RBF) interpolant.
    
    Strategy:
    - Use 10 fixed centers in the normalized 4D input space.
    - The 10 parameters are the weights for these RBFs.
    - This is a linear-in-parameters model, making it convex and easy for BFGS to optimize.
    - Fixed centers: 4 corners of the hypercube, 4 edge midpoints (representative), and 2 others (centroid, and one more spread out).
    
    Inputs:
    - b: biomass, range ~[1, 5]
    - s: substrate, range ~[1, 5]
    - temp: temperature, range ~[1, 40]
    - pH: pH, range ~[2, 12]
    
    Normalization:
    - b_norm = (b - 1) / 4
    - s_norm = (s - 1) / 4
    - t_norm = (temp - 1) / 39
    - p_norm = (pH - 2) / 10
    
    Centers (in normalized space [0,1]^4):
    1. [0, 0, 0, 0]
    2. [1, 1, 1, 1]
    3. [0.5, 0.5, 0.5, 0.5] (centroid)
    4. [0, 0, 1, 1]
    5. [1, 1, 0, 0]
    6. [0, 1, 0, 1]
    7. [1, 0, 1, 0]
    8. [0.5, 0, 0.5, 1]
    9. [0.5, 1, 0.5, 0]
    10. [0, 0.5, 1, 0.5]
    
    Kernel: Gaussian with fixed width.
    Width selection: 
    Since inputs are normalized to [0,1], a width of ~0.3-0.5 is reasonable.
    Let's use sigma = 0.4.
    """
    # Normalize inputs
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    p_norm = (pH - 2.0) / 10.0
    
    # Stack normalized inputs into a matrix of shape (N, 4)
    X = np.column_stack((b_norm, s_norm, t_norm, p_norm))
    
    # Define 10 fixed centers
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5, 1.0],
        [0.5, 1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0, 0.5]
    ])
    
    # Compute squared distances from each point to each center
    # X: (N, 4), centers: (10, 4)
    # diff: (N, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)  # (N, 10)
    
    # Gaussian kernel with fixed sigma
    sigma = 0.4
    weights = params  # (10,)
    
    # Compute RBF values
    # K: (N, 10)
    K = np.exp(-sq_dist / (2.0 * sigma**2))
    
    # Linear combination
    y = K @ weights
    
    return y
