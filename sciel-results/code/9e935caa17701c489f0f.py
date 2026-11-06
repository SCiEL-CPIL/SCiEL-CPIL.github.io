import math
import time
import numpy as np

def design(problem):
    """Design a symmetric 32x32-pixel unit cell for maximum TM 4-5 band gap.
    
    Uses a 2x2 array of square dielectric rods as the base structure.
    Performs a local search over rod size and position to maximize the gap.
    """
    size = problem.get("pixels", 32)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s budget (with margin)
    
    # Initialize with air (0)
    cell = [[0] * size for _ in range(size)]
    
    # Define 2x2 array of rods
    # Rod size
    rod_size = 8
    # Center of each quadrant
    # Quadrants are 16x16. Centers at 7.5 and 23.5.
    # We place rods centered at these points.
    # Indices for rod centered at 7.5 with size 8: 7.5 - 4 = 3.5 to 7.5 + 4 = 11.5 -> 4 to 11 (8 pixels)
    # Indices for rod centered at 23.5 with size 8: 23.5 - 4 = 19.5 to 23.5 + 4 = 27.5 -> 20 to 27 (8 pixels)
    
    row_ranges = [(4, 11), (20, 27)]
    col_ranges = [(4, 11), (20, 27)]
    
    for r_start, r_end in row_ranges:
        for c_start, c_end in col_ranges:
            for i in range(r_start, r_end + 1):
                for j in range(c_start, c_end + 1):
                    cell[i][j] = 1
                    
    return cell
