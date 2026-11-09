import math

def design(problem):
    size = problem["pixels"]
    cell = []
    # Define 1 large air hole centered at (0,0) in a dielectric background
    # This is the inverse topology of the seed (which had dielectric rods in air)
    # We use a single central feature to maximize the gap for higher bands
    r = 0.35
    center_x = 0.5
    center_y = 0.5
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size
            y = (j + 0.5) / size
            dx = x - center_x
            dy = y - center_y
            # If inside the hole, it is air (0), otherwise dielectric (1)
            if dx * dx + dy * dy < r * r:
                is_dielectric = 0
            else:
                is_dielectric = 1
            row.append(is_dielectric)
        cell.append(row)
    return cell
