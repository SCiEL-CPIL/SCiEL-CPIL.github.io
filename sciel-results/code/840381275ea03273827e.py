import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = x.flatten()
    y = y.flatten()
    
    # Define 10 fixed centers in the [-1, 1] x [-1, 1] domain
    # 4 corners, 4 mid-edges (or mid-squares), 2 vertical midpoints?
    # Let's use a 3x3 grid (9 pts) + 1 center at (0,0) is redundant if grid has 0,0.
    # Grid: -1, 0, 1 for x and y.
    # Points:
    # (-1,-1), (-1,0), (-1,1)
    # (0,-1), (0,0), (0,1)
    # (1,-1), (1,0), (1,1)
    # This is 9 points.
    # Add (0, 0.5) or similar?
    # Let's add (0.5, 0.5) and remove (0,0)? No, (0,0) is important.
    # Let's use 10 points:
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0],
        [ 0.0,  0.5]  # 10th point
    ])
    
    # Compute squared distances from each (x, y) to each center
    # x: (N,), centers: (10, 2)
    # dist_sq: (N, 10)
    # (x - cx)^2 + (y - cy)^2
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Gaussian kernel
    gamma = 2.0  # Fixed width parameter
    features = np.exp(-gamma * dist_sq)
    
    # Linear combination
    return features @ params
