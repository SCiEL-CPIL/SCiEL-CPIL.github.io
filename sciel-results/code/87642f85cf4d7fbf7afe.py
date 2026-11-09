import math

def design(problem):
    size = problem["pixels"]
    cell = []
    # Define 4 air holes centered at (+/-0.25, +/-0.25) in a dielectric background
    # Radius r = 0.15
    r = 0.15
    centers = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            is_dielectric = 1  # Background is dielectric
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < r * r:
                    is_dielectric = 0  # Hole is air
                    break
            row.append(is_dielectric)
        cell.append(row)
    return cell
