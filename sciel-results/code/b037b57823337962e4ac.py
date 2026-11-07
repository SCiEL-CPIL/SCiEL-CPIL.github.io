import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Flatten inputs to 1D for easy broadcasting
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    p = params.flatten()
    
    # Define centers
    # 8 corners of [0.5, 2.0]^3
    c1 = np.array([0.5, 0.5, 0.5])
    c2 = np.array([2.0, 0.5, 0.5])
    c3 = np.array([0.5, 2.0, 0.5])
    c4 = np.array([2.0, 2.0, 0.5])
    c5 = np.array([0.5, 0.5, 2.0])
    c6 = np.array([2.0, 0.5, 2.0])
    c7 = np.array([0.5, 2.0, 2.0])
    c8 = np.array([2.0, 2.0, 2.0])
    # 1 center
    c9 = np.array([1.25, 1.25, 1.25])
    
    # Stack centers into a (9, 3) array
    centers = np.vstack([c1, c2, c3, c4, c5, c6, c7, c8, c9])
    
    # Stack inputs into a (N, 3) array
    inputs = np.column_stack((x, y, z))
    
    # Compute squared distances: (N, 1, 3) - (1, 9, 3) -> (N, 9, 3) -> sum -> (N, 9)
    # This might be memory heavy if N is large, but N=200 is small.
    # Alternative: loop over centers to save memory and be explicit.
    # Given N=200 and 9 centers, (200, 9, 3) is 5400 floats, very small.
    
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    
    # Width sigma
    sigma = 0.7
    # Gaussian basis values
    basis = np.exp(-sq_dist / (2 * sigma**2))
    
    # Add constant basis term
    # params[0..8] for RBFs, params[9] for constant
    constant_term = p[9]
    
    # Prediction
    pred = basis @ p[:9] + constant_term
    
    return pred
