import math

def design(problem):
    """
    Designs a 32x32 unit cell with a superellipse dielectric rod.
    Uses C4v symmetry.
    Parameters: R=0.31, alpha=4.0 (Squarish rod)
    """
    size = problem["pixels"]
    
    # Superellipse parameters
    R = 0.31
    alpha = 4.0
    R_alpha = R ** alpha
    
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Iterate over all pixels
    for i in range(size):
        for j in range(size):
            # Calculate pixel center coordinates in [-0.5, 0.5]
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Check superellipse condition
            # |x|^alpha + |y|^alpha <= R^alpha
            if abs(x) ** alpha + abs(y) ** alpha <= R_alpha:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
                
    return cell
