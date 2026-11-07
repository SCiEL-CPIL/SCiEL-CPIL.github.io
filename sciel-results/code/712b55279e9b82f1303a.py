import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using fixed Gaussian RBF basis.
    
    Uses 10 Gaussian kernels centered at fixed locations in [-1,1]^2.
    Each basis function is exp(-||[x,y] - c_k||^2 / (2*sigma^2)).
    This is well-conditioned for BFGS and can approximate smooth functions.
    """
    # Fixed centers spread across [-1,1]^2
    centers = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0,  1.0],
        [ 1.0,  1.0],
        [ 0.0,  0.0],  # duplicate center with different effective weight
    ])
    
    sigma = 0.5
    
    # Compute distances from each point to each center
    # x, y are arrays of shape (n,)
    # centers is (10, 2)
    # We want dist_sq[i, k] = (x[i] - centers[k,0])^2 + (y[i] - centers[k,1])^2
    
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # (n, 10)
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # (n, 10)
    dist_sq = dx * dx + dy * dy  # (n, 10)
    
    # Gaussian RBF: exp(-dist_sq / (2 * sigma^2))
    basis = np.exp(-dist_sq / (2.0 * sigma * sigma))  # (n, 10)
    
    # Linear combination: f = sum_k params[k] * basis[i, k]
    f = basis @ params
    
    return f
