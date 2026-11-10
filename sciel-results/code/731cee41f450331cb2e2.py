"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """A centered square dielectric of side 0.5 lattice constants."""
    size = problem["pixels"]
    cell = []
    half_size = 0.25  # Side is 0.5, so half-side is 0.25
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            # Check if inside the square
            if abs(x) < half_size and abs(y) < half_size:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
