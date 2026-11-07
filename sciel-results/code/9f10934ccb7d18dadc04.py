"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF interpolant with 10 fixed centers in normalized input space.
    Centers: centroid, 8 axis-extremes (face centers), and one corner.
    Gaussian kernels with fixed bandwidth sigma=0.7.
    Linear-in-parameters: prediction = sum_i params[i] * exp(-||x-c_i||^2/(2*sigma^2))
    """
    # Normalize inputs to roughly [-1, 1]
    b_norm = (b - 2.5) / 2.0
    s_norm = (s - 20.0) / 20.0
    temp_norm = (temp - 20.0) / 10.0
    pH_norm = (pH - 7.0) / 5.0

    # Stack into (n, 4)
    X = np.column_stack([b_norm, s_norm, temp_norm, pH_norm])

    # 10 fixed centers
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],   # 0: centroid
        [0.0, 1.0, 0.0, 0.0],   # 1: s max
        [0.0, -1.0, 0.0, 0.0],  # 2: s min
        [0.0, 0.0, 1.0, 0.0],   # 3: temp max
        [0.0, 0.0, -1.0, 0.0],  # 4: temp min
        [0.0, 0.0, 0.0, 1.0],   # 5: pH max
        [0.0, 0.0, 0.0, -1.0],  # 6: pH min
        [1.0, 0.0, 0.0, 0.0],   # 7: b max
        [-1.0, 0.0, 0.0, 0.0],  # 8: b min
        [1.0, 1.0, 1.0, 1.0],   # 9: far corner (all max)
    ])

    sigma = 0.7

    # Compute squared distances: (n, 10)
    # Using broadcasting: X (n,1,4), centers (1,10,4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)

    # RBF values
    phi = np.exp(-sq_dists / (2.0 * sigma * sigma))  # (n, 10)

    # Weighted sum
    mu = phi @ params

    return mu
# EVOLVE-BLOCK-END
