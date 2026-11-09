"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Strategy: Gaussian RBF Network with 10 fixed centers.
    # The 10 parameters are the amplitudes of the RBFs.
    # This avoids the conditioning issues of high-order polynomials and can model non-polynomial shapes.
    
    # Define 10 centers evenly distributed in the domain [0.5, 2.0]^3.
    # We use a 2x2x2 grid (8 points) plus the center of the domain (1 point) and one corner (1 point)
    # to ensure good coverage and distinctiveness.
    
    # Grid points:
    # x_coords: [0.5, 2.0]
    # y_coords: [0.5, 2.0]
    # z_coords: [0.5, 2.0]
    
    # Let's pick specific centers to ensure they are distinct and spread out.
    # Center 0: (1.25, 1.25, 1.25) - Center of domain
    # Centers 1-8: Corners of the cube [0.5, 2.0]^3
    # Center 9: (1.25, 1.25, 0.5) - Midpoint of bottom face? Or maybe (0.5, 1.25, 1.25)
    
    centers = np.array([
        [1.25, 1.25, 1.25],  # 0: Center
        [0.5,  0.5,  0.5],   # 1: Corner
        [2.0,  0.5,  0.5],   # 2: Corner
        [0.5,  2.0,  0.5],   # 3: Corner
        [2.0,  2.0,  0.5],   # 4: Corner
        [0.5,  0.5,  2.0],   # 5: Corner
        [2.0,  0.5,  2.0],   # 6: Corner
        [0.5,  2.0,  2.0],   # 7: Corner
        [2.0,  2.0,  2.0],   # 8: Corner
        [1.25, 1.25, 0.5]    # 9: Midpoint of one face to add density
    ])
    
    # Bandwidth (sigma) for the Gaussian kernels.
    # The domain size is 1.5 in each dimension.
    # A sigma of 0.5-0.8 seems reasonable to allow overlap but keep locality.
    # If sigma is too small, we get interpolation issues. If too large, we lose locality.
    # Let's try sigma = 0.6.
    sigma = 0.6
    gamma = 1.0 / (2.0 * sigma * sigma)
    
    # Compute distances from each point (x,y,z) to each center.
    # x, y, z are arrays of shape (N,)
    # We need to compute the kernel matrix K of shape (N, 10)
    
    # To avoid creating huge intermediate arrays if N is large, we can loop over centers,
    # but N=200 is small, so vectorization is fine and fast.
    
    # Reshape inputs to (N, 1)
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    z_col = z[:, np.newaxis]
    
    # Centers are (10, 3)
    # Compute squared distances: (N, 1, 3) - (1, 10, 3) -> (N, 10, 3) -> sum over axis 2 -> (N, 10)
    
    # dx = x_col - centers[:, 0][np.newaxis, :]  # (N, 10)
    # dy = y_col - centers[:, 1][np.newaxis, :]  # (N, 10)
    # dz = z_col - centers[:, 2][np.newaxis, :]  # (N, 10)
    
    # dist_sq = dx**2 + dy**2 + dz**2
    
    # More efficient:
    # x2 = x_col**2 + centers[:, 0]**2  # (N, 1) + (10,) -> (N, 10)
    # ...
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    cz = centers[:, 2]
    
    dx = x_col - cx[np.newaxis, :]
    dy = y_col - cy[np.newaxis, :]
    dz = z_col - cz[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2 + dz**2
    
    # Compute Gaussian kernels: exp(-gamma * dist_sq)
    K = np.exp(-gamma * dist_sq)
    
    # The model is K @ params
    # params is shape (10,)
    # K is shape (N, 10)
    # Result is shape (N,)
    
    return K @ params
# EVOLVE-BLOCK-END
