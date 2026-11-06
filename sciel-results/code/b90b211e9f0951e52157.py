"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    A four-rod lattice with rods centered at (±0.25, ±0.25).
    This symmetry breaks the vanishing of the (1,1) Fourier component,
    which is essential for opening the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Define rod centers in normalized coordinates [0, 1)
    # Centers at (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
    # This ensures 4-fold rotational symmetry and inversion symmetry about (0.5, 0.5)
    centers = [
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75)
    ]
    
    # Radius chosen to be substantial but not overlapping.
    # Distance between nearest centers is 0.5.
    # Radius 0.18 gives fill factor approx 4 * pi * 0.18^2 = 0.407.
    # Higher fill factor often improves higher-order gaps for high-index dielectrics.
    radius = 0.18
    
    for i in range(size):
        # Pixel center y
        y = (i + 0.5) / size
        for j in range(size):
            # Pixel center x
            x = (j + 0.5) / size
            
            # Check if pixel is inside any of the rods
            is_dielectric = False
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < radius * radius:
                    is_dielectric = True
                    break
            
            if is_dielectric:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
                
    return cell
# EVOLVE-BLOCK-END
