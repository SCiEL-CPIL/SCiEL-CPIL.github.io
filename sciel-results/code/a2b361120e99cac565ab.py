import time
import numpy as np

def design(problem):
    size = problem["pixels"]
    
    # Cross-shaped dielectric: strong (1,0) and (0,1) Fourier components
    # This should enhance coupling between bands with different symmetry
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Cross: |x| < 0.175 or |y| < 0.175
            if abs(x) < 0.175 or abs(y) < 0.175:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    
    # Enforce full square symmetry
    # 180° rotation
    for i in range(size):
        for j in range(size):
            cell[i][j] = min(cell[i][j], cell[size-1-i][size-1-j])
    # Horizontal reflection
    for i in range(size):
        for j in range(size):
            cell[i][j] = min(cell[i][j], cell[i][size-1-j])
    # Vertical reflection
    for i in range(size):
        for j in range(size):
            cell[i][j] = min(cell[i][j], cell[size-1-i][j])
    # Transpose
    for i in range(size):
        for j in range(size):
            cell[i][j] = min(cell[i][j], cell[j][i])
    
    return cell
