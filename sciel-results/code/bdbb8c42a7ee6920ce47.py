import math

def design(problem):
    size = problem["pixels"]
    # We assume size is 32
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Centers of the holes
    # Center of the cell
    cx = size / 2.0
    cy = size / 2.0
    
    # Radius for the holes
    # For a centered square lattice with lattice constant a = size/sqrt(2) ~ 22.6
    # Optimal r/a is roughly 0.35-0.4
    # r ~ 8-9
    radius = 8.5
    
    # Define hole centers
    # The centered square lattice in this 32x32 supercell has holes at:
    # 1. The center (16, 16)
    # 2. The four corners (0,0), (32,0), (0,32), (32,32)
    # Note: (32,0) is equivalent to (0,0) in periodic boundary conditions, 
    # but since we are drawing pixels, we check distance to all these points.
    
    hole_centers = [
        (cx, cy),
        (0, 0),
        (size, 0),
        (0, size),
        (size, size)
    ]
    
    for i in range(size):
        for j in range(size):
            # Pixel center coordinates
            x = i + 0.5
            y = j + 0.5
            
            is_air = False
            for hx, hy in hole_centers:
                dist_sq = (x - hx)**2 + (y - hy)**2
                if dist_sq < radius**2:
                    is_air = True
                    break
            
            if is_air:
                cell[i][j] = 0
            else:
                cell[i][j] = 1
                
    return cell
