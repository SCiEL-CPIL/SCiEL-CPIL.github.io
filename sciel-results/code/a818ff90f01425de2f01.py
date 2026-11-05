"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """Dielectric background with a central air hole, optimized for TM band gap."""
    size = problem["pixels"]
    cell = []
    r = 0.32  # air hole radius in lattice constants
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Air hole: 0 if inside circle, dielectric (1) otherwise
            if x * x + y * y < r * r:
                row.append(0)
            else:
                row.append(1)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
