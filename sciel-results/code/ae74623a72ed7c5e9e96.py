"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between TM bands 4 and 5.
    
    Strategy: Geometric Primitive Construction.
    Instead of Fourier synthesis, we define the dielectric distribution using
    explicit geometric shapes (squares/circles) that are known to support
    large band gaps in square lattices.
    
    We construct a pattern with a dielectric background and air holes,
    or dielectric pillars in air, using high-symmetry configurations.
    
    We will implement a "Holey Square" design:
    1. Start with a full dielectric background (all 1s).
    2. Punch out a central air hole (square).
    3. Punch out four corner air holes (squares).
    
    This creates a high-contrast structure with strong scattering centers.
    """
    size = problem["pixels"]
    
    # Initialize with all 1s (Dielectric)
    cell_np = np.ones((size, size), dtype=int)
    
    # Define the central air hole
    # A square hole in the center.
    # Size: 12x12 pixels.
    # Center is at (15.5, 15.5).
    # So indices 10 to 21 (inclusive) cover 12 pixels centered on 15.5?
    # 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21.
    # Center of this range: (10+21)/2 = 15.5. Correct.
    hole_size_center = 12
    start_c = (size - hole_size_center) // 2
    end_c = start_c + hole_size_center
    cell_np[start_c:end_c, start_c:end_c] = 0
    
    # Define the corner air holes
    # Four squares in the corners.
    # Size: 8x8 pixels.
    # Top-Left: rows 0-7, cols 0-7
    # Top-Right: rows 0-7, cols 24-31
    # Bottom-Left: rows 24-31, cols 0-7
    # Bottom-Right: rows 24-31, cols 24-31
    hole_size_corner = 8
    cell_np[0:hole_size_corner, 0:hole_size_corner] = 0
    cell_np[0:hole_size_corner, size-hole_size_corner:size] = 0
    cell_np[size-hole_size_corner:size, 0:hole_size_corner] = 0
    cell_np[size-hole_size_corner:size, size-hole_size_corner:size] = 0
    
    # Ensure strict symmetry (should be inherently symmetric, but enforce just in case)
    # The construction is symmetric under 180 rotation and transpose.
    
    # Convert to list of lists
    cell = cell_np.tolist()
    
    return cell
# EVOLVE-BLOCK-END
