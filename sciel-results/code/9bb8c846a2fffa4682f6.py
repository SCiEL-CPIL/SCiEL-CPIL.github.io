import numpy as np

def squares_in_square10():
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    # 8 squares of side 1/3
    s_large = 1.0 / 3.0
    # 2 squares of side 1/6
    s_small = 1.0 / 6.0
    
    # Top 2 rows (6 squares)
    # Row 1 (y = 5/6)
    centers[0] = [1/6, 5/6]; sides[0] = s_large
    centers[1] = [1/2, 5/6]; sides[1] = s_large
    centers[2] = [5/6, 5/6]; sides[2] = s_large
    
    # Row 2 (y = 1/2)
    centers[3] = [1/6, 1/2]; sides[3] = s_large
    centers[4] = [1/2, 1/2]; sides[4] = s_large
    centers[5] = [5/6, 1/2]; sides[5] = s_large
    
    # Bottom row, left 2 large squares
    # Row 3 (y = 1/6)
    centers[6] = [1/6, 1/6]; sides[6] = s_large
    centers[7] = [1/2, 1/6]; sides[7] = s_large
    
    # Bottom row, right 2 small squares
    # They fit in x=[2/3, 1], y=[0, 1/3]
    # Place them side-by-side at the bottom
    # Square 8: x center 5/6, y center 1/12
    centers[8] = [5/6, 1/12]; sides[8] = s_small
    # Square 9: x center 11/12, y center 1/12
    centers[9] = [11/12, 1/12]; sides[9] = s_small
    
    return centers, sides
