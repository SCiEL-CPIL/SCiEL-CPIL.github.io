"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between TM bands 4 and 5.
    
    Strategy: Use a dielectric ring with a square (Chebyshev distance) cross-section.
    This approach uses a different geometric norm (L-infinity) compared to the 
    diamond (L-1) approach, potentially coupling more effectively with the 
    square lattice symmetry and plane-wave components.
    
    The cell must be symmetric under:
    - 180-degree rotation (cell == cell[::-1] row-wise and column-wise)
    - Transpose symmetry (cell == cell.T)
    
    We design a centered square annulus using Chebyshev distance.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    cell = [[0] * size for _ in range(size)]
    
    # Center of the 32x32 grid
    center = (size - 1) / 2.0
    
    # Parameters for the square annulus
    # Using Chebyshev distance: max(|x - cx|, |y - cy|)
    # Tuned radii to target high filling fraction and strong Bragg scattering
    r_inner = 0.22 * size  # ~7.0 pixels
    r_outer = 0.42 * size  # ~13.4 pixels
    
    for i in range(size):
        for j in range(size):
            # Chebyshev distance from center
            d = max(abs(i - center), abs(j - center))
            
            if r_inner <= d <= r_outer:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
    
    return cell
# EVOLVE-BLOCK-END
