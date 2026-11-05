"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """A 5-rod cluster with tuned parameters for higher-order TM band gap."""
    size = problem["pixels"]
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    
    r_center = 3.0
    dist = 4.5
    r_surround = 2.5
    
    centers = [
        (cx, cy, r_center),
        (cx - dist, cy, r_surround),
        (cx + dist, cy, r_surround),
        (cx, cy - dist, r_surround),
        (cx, cy + dist, r_surround)
    ]
    
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for (px, py, r) in centers:
                if (x - px)**2 + (y - py)**2 < r**2:
                    cell[i][j] = 1
                    break
    return cell
# EVOLVE-BLOCK-END
