import math

def design(problem):
    size = problem["pixels"]
    cell = []
    # Define a central square air hole in a dielectric background
    # Half-width of the square hole
    r = 0.35
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Start with dielectric (1)
            is_dielectric = 1
            # Check if inside the square hole
            if abs(x) < r and abs(y) < r:
                is_dielectric = 0
            row.append(is_dielectric)
        cell.append(row)
    return cell
