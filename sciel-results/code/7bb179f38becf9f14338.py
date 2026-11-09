import math

def design(problem):
    size = problem["pixels"]
    cell = []
    r_inner = 0.13
    r_outer = 0.37
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x * x + y * y
            row.append(1 if (r_inner * r_inner < r2 < r_outer * r_outer) else 0)
        cell.append(row)
    return cell
