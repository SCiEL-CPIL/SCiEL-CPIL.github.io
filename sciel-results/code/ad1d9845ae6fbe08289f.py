"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """
    A centered square dielectric rod.
    Square rods have stronger Fourier components at the primary reciprocal lattice vectors
    compared to circular rods, enhancing Bragg scattering and band gaps.
    """
    size = problem["pixels"]
    cell = []
    
    # Center of the 32x32 grid is between pixels 15 and 16 (15.5)
    center = 15.5
    
    # Half-width of the square rod
    # Side length = 2 * half_width
    # We choose half_width = 7.0, resulting in a 14x14 pixel square.
    # This covers pixels 9 to 22 inclusive.
    half_width = 7.0
    
    for i in range(size):
        row = []
        for j in range(size):
            # Calculate distance from center in pixel units
            dx = abs(i - center)
            dy = abs(j - center)
            
            # Check if pixel is inside the square
            if dx < half_width and dy < half_width:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
        
    return cell
# EVOLVE-BLOCK-END
