import numpy as np

def equation(x, y, params):
    # Centers for the Gaussian basis functions
    # 9 local centers + 1 global center
    centers = np.array([
        [-0.8, -0.8],
        [ 0.0, -0.8],
        [ 0.8, -0.8],
        [-0.8,  0.0],
        [ 0.0,  0.0],
        [ 0.8,  0.0],
        [-0.8,  0.8],
        [ 0.0,  0.8],
        [ 0.8,  0.8],
        [ 0.0,  0.0]  # Global center
    ])
    
    # Widths (sigma) for each Gaussian
    # Local centers have narrower width, global center has wider width
    sigmas = np.array([
        0.6, 0.6, 0.6,
        0.6, 0.6, 0.6,
        0.6, 0.6, 0.6,
        1.5
    ])
    
    # Compute squared distances from each point (x, y) to each center
    # x and y are 1D arrays of length N
    # centers is 10x2
    # We want a matrix of shape (N, 10)
    
    # x shape (N,), y shape (N,)
    # centers[:, 0] shape (10,), centers[:, 1] shape (10,)
    
    dx = x[:, np.newaxis] - centers[0, 0] # This is wrong, need broadcasting
    # Correct broadcasting:
    # x (N, 1), centers[0] (10, 1) -> dx (N, 10)
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    # Compute Gaussian values
    # sigma (10, 1)
    # dist_sq (N, 10)
    # result (N, 10)
    sigmas_sq = (sigmas[:, np.newaxis]**2)
    
    # Avoid division by zero or underflow issues, though sigmas are fixed > 0
    gaussians = np.exp(-dist_sq / (2 * sigmas_sq))
    
    # Linear combination
    # gaussians (N, 10), params (10,)
    # result (N,)
    return gaussians @ params
