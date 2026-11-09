import math

def design(problem):
    size = problem["pixels"]
    cell = []
    half = size // 2
    for i in range(size):
        row = []
        for j in range(size):
            # Checkerboard of 4 quadrants
            if (i < half and j < half) or (i >= half and j >= half):
                is_dielectric = 1
            else:
                is_dielectric = 0
            row.append(is_dielectric)
        cell.append(row)
    return cell
