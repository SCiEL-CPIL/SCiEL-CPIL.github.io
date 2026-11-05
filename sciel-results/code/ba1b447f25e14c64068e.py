"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    A centered dielectric rod with a superellipse (squircle) shape.
    This generalizes the circular rod (n=2) and approaches a square (n=infinity).
    For higher-order TM bands in square lattices, the flatter edges of a squircle
    often help shift band edges to open larger gaps compared to a pure circle.
    """
    size = problem["pixels"]
    # Parameters
    # Radius equivalent (half-width)
    r = 0.22 
    # Exponent for superellipse: 2 is circle, 4 is squircle, 10 is near-square
    n = 4.0
    
    cell = []
    # Center is at (size/2 - 0.5, size/2 - 0.5) in pixel coordinates
    # We use normalized coordinates x, y in [-0.5, 0.5]
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            # Superellipse equation: |x/r|^n + |y/r|^n <= 1
            # Avoid division by zero if r is 0, but r is fixed
            val = abs(x/r)**n + abs(y/r)**n
            if val <= 1.0:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
# EVOLVE-BLOCK-END
