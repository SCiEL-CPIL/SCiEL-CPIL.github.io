import time
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    size = problem["pixels"]
    # Initialize with 0
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Define 4 rod centers
    # 2x2 superlattice of rods
    # Centers at (size/4, size/4), (size/4, 3*size/4), (3*size/4, size/4), (3*size/4, 3*size/4)
    centers = [
        (size // 4, size // 4),
        (size // 4, 3 * size // 4),
        (3 * size // 4, size // 4),
        (3 * size // 4, 3 * size // 4)
    ]
    
    # Radius: start with 6 pixels (diameter 12)
    # Distance between centers is 16. Radius 6 leaves 4px gap.
    radius = 6.0
    
    # Fill the rods
    for i in range(size):
        for j in range(size):
            for cx, cy in centers:
                dx = i - cx
                dy = j - cy
                if dx * dx + dy * dy <= radius * radius:
                    cell[i][j] = 1
                    break
            
    # Check symmetry
    # The construction is symmetric by design.
    
    return cell
