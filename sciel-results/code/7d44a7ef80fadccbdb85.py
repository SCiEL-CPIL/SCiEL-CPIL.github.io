"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Gaussian RBF basis.
    
    Strategy: Use a fixed grid of Gaussian kernels centered in the normalized 
    (x, v) space. This provides a flexible, local basis that can approximate 
    arbitrary smooth nonlinear functions better than a global polynomial of 
    the same parameter count. The 10 parameters correspond to the weights of 
    10 RBFs placed at specific locations covering the training domain.
    """
    # Normalize inputs based on training ranges provided in the prompt
    # x range: [-0.5728, 0.5275] -> scale by max abs ~0.57
    # v range: [-0.1754, 0.2821] -> scale by max abs ~0.28
    x_norm = x / 0.57
    v_norm = v / 0.28
    
    # Define 10 RBF centers in the normalized (x, v) space.
    # We distribute them to cover the likely active region of the oscillator.
    # The oscillator likely stays within a bounded region (limit cycle).
    # Let's place centers in a rough grid covering [-1, 1] x [-1, 1] roughly.
    
    # Centers (x_c, v_c)
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
        [ 0.0,  0.0] # Duplicate center? No, let's make it distinct or use a different spread.
                     # Actually, having a duplicate center with different width isn't possible with fixed centers.
                     # Let's adjust the last center to be somewhere else, e.g., corner or edge.
                     # Or just use a 3x3 grid + 1 extra? 
                     # Let's stick to a 3x3 grid (9 points) + 1 point at origin? 
                     # Wait, the origin is already in the 3x3 grid.
                     # Let's use a 3x3 grid centered at 0.
                     # Points: -0.8, 0, 0.8 for both x and v.
                     # That's 9 points. We have 10 params.
                     # Let's add a 10th point at a specific location, e.g., (0.5, 0.5) or just repeat one?
                     # Repeating a center effectively doubles its weight if params are independent? No, they are summed.
                     # If center 4 and 9 are identical, p[4]*G + p[9]*G = (p[4]+p[9])*G. 
                     # This reduces effective degrees of freedom. 
                     # Better to have 10 distinct centers.
                     # Let's use a 4x2 grid? Or 3x3 + 1?
                     # Let's use 3x3 grid:
                     # (-0.8, -0.8), (0, -0.8), (0.8, -0.8)
                     # (-0.8, 0),   (0, 0),   (0.8, 0)
                     # (-0.8, 0.8), (0, 0.8), (0.8, 0.8)
                     # And add one more at (0.0, -0.4) or similar?
                     # Or just use a slightly denser grid?
                     # Let's try a 5x2 grid? No, 10 params.
                     # Let's use a 3x3 grid (9 points) and one point at (0.0, 0.0) is redundant.
                     # Let's replace the center (0,0) with two points? No.
                     # Let's just use 10 distinct points.
                     # Grid:
                     # x: -0.8, -0.27, 0.27, 0.8
                     # v: -0.8, 0.0, 0.8
                     # That's 12 points. Too many.
                     
                     # Let's stick to the 3x3 grid and add one point at (0.0, 0.0) is bad.
                     # How about a 4x2 grid?
                     # x: -0.8, -0.27, 0.27, 0.8
                     # v: -0.5, 0.5
                     # 4 * 2 = 8 points. Need 2 more.
                     # v: 0.0
                     # 4 * 3 = 12 points.
                     
                     # Let's just use a 3x3 grid and one point at (0.0, 0.0) is the center.
                     # Wait, the 3x3 grid includes (0,0).
                     # So we have 9 unique points.
                     # Let's add a 10th point at (0.0, 0.0) with a different width? 
                     # We can't vary width per param easily without extra params.
                     # But we can define fixed widths.
                     # If we have two RBFs at the same center with different widths, that's fine.
                     # Let's do that.
                     # Center 4: (0,0) width 0.5
                     # Center 9: (0,0) width 1.5
                     # This allows modeling both sharp and broad features at the origin.
        
    ])
    
    # Let's redefine centers clearly for 10 RBFs
    # 3x3 grid
    xs = [-0.8, 0.0, 0.8]
    vs = [-0.8, 0.0, 0.8]
    centers = []
    for vx in vs:
        for vx in xs: # Wait, nested loop order doesn't matter for list
            centers.append([vx, vx]) # Typo in loop variable above, let's be explicit
    
    # Explicit list for clarity and control
    centers = [
        (-0.8, -0.8),
        ( 0.0, -0.8),
        ( 0.8, -0.8),
        (-0.8,  0.0),
        ( 0.0,  0.0),
        ( 0.8,  0.0),
        (-0.8,  0.8),
        ( 0.0,  0.8),
        ( 0.8,  0.8),
        ( 0.0,  0.0)  # 10th RBF at origin, will use different width
    ]
    
    centers = np.array(centers)
    
    # Define fixed widths (sigmas) for the Gaussians.
    # Smaller sigma = sharper peak.
    # Let's use a standard sigma for the grid points and a larger one for the duplicate origin.
    sigmas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
    
    # Calculate squared distances from each point to each center
    # x_norm, v_norm are (N, 1) or (N,). 
    # Centers are (10, 2).
    
    # To avoid memory blowup for large N, we can compute in chunks or rely on NumPy broadcasting efficiency.
    # N=10000, K=10. 10000x10 matrix of distances is fine.
    
    dx = x_norm[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # (N, 10)
    dv = v_norm[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # (N, 10)
    
    sq_dist = dx**2 + dv**2  # (N, 10)
    
    # Apply widths
    # sigmas is (10,). We need to divide by sigmas^2.
    sq_dist = sq_dist / (sigmas[np.newaxis, :]**2)
    
    # Gaussian kernel: exp(-0.5 * sq_dist)
    # Note: Standard Gaussian is exp(-x^2 / (2 * sigma^2)).
    # So we divide by 2 * sigma^2.
    sq_dist = sq_dist / 2.0
    
    phi = np.exp(-sq_dist)  # (N, 10)
    
    # Linear combination with params
    # params is (10,)
    # result is (N,)
    return phi @ params

# EVOLVE-BLOCK-END
