"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between TM bands 4 and 5.
    
    Strategy: Geometric Primitive (Central Circular Air Hole).
    This approach departs from the Fourier synthesis method by directly
    defining a geometric shape. We use a circular air hole (epsilon=1)
    centered in a dielectric background (epsilon=12).
    
    The square lattice symmetry is naturally preserved by the central
    circular geometry. The radius is chosen to provide a filling fraction
    that typically opens a significant band gap in the lower bands for
    TM polarization with high index contrast.
    """
    size = problem["pixels"]
    
    # Initialize the cell with Dielectric (1)
    cell_np = np.ones((size, size), dtype=int)
    
    # Define the center of the cell
    center = (size - 1) / 2.0
    
    # Create coordinate grids
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from center for each pixel
    # Using pixel centers: x=0 is at 0, x=31 is at 31. Center is 15.5.
    dist_sq = (X - center)**2 + (Y - center)**2
    
    # Define the radius of the air hole.
    # A radius of ~11-12 pixels gives a hole area of ~380-450 pixels.
    # Total area is 1024.
    # Filling fraction of hole: ~0.37 - 0.44.
    # This is a standard range for optimal gaps in square lattices.
    radius = 11.5
    radius_sq = radius**2
    
    # Set pixels inside the radius to Air (0)
    # Using <= to include the boundary pixels that are closer to the center
    hole_mask = dist_sq <= radius_sq
    cell_np[hole_mask] = 0
    
    # Convert to list of lists
    cell = cell_np.tolist()
    
    return cell
# EVOLVE-BLOCK-END
