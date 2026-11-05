"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell for a 2D square-lattice photonic crystal
    with epsilon_high=12, aiming to maximize the TM band gap between bands 4 and 5.
    
    Approach: Use a hole-pattern design. The square symmetry group requires the cell
    to be invariant under 90-degree rotation and reflection about the center.
    We design a pattern of dielectric rods (or equivalently, air holes in a dielectric background)
    that is known to produce large TM band gaps.
    
    For TM polarization in square lattice, the best-known designs often use:
    - A central hole or rod
    - Four additional holes/rods at positions related by 4-fold symmetry
    
    Let's try a pattern with a central dielectric region surrounded by air holes,
    or vice versa. Given epsilon_high=12 (high contrast), we want to maximize
    the photonic band gap.
    
    Strategy: Start with a known good configuration for square lattice TM gap.
    The classic design for large TM gaps in square lattice uses air holes in a
    dielectric background. Let's use a pattern with:
    - Central dielectric dot
    - Four dielectric dots at (±a, 0) and (0, ±a) positions
    This creates a cross-like pattern that breaks the symmetry appropriately.
    
    Actually, let me think more carefully. The symmetry requirement is:
    cell == cell[::-1] row-wise (vertical flip)
    cell == cell[::-1] column-wise (horizontal flip) 
    cell == cell.T (transpose, i.e., 90-degree rotation symmetry)
    
    So we need 4-fold rotational symmetry plus reflections.
    
    Let me design a pattern with:
    - A central dielectric region (small)
    - Four dielectric regions at the midpoints of the edges
    - Or: air holes at specific positions
    
    For maximum TM band gap in square lattice with high epsilon contrast,
    research suggests using a pattern with a central hole and four corner holes,
    or a pattern that approximates a circle with additional features.
    
    Let me try: dielectric background with air holes at:
    - Center (0,0) in normalized coords
    - (±0.25, 0), (0, ±0.25)
    This gives 5 holes total with 4-fold symmetry.
    
    Alternatively: air background with dielectric rods at:
    - Center
    - (±0.3, 0), (0, ±0.3)
    
    Let me try the first approach: dielectric background (all 1s) with air holes.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Create a 32x32 cell
    # We'll use a pattern with air holes in a dielectric background
    # Holes at center and at four positions along axes
    
    # Normalize coordinates: pixel (i,j) has center at ((i+0.5)/size - 0.5, (j+0.5)/size - 0.5)
    # Range: -0.5 to 0.5
    
    # Define hole centers (in normalized coords, where cell spans -0.5 to 0.5)
    hole_centers = [
        (0.0, 0.0),       # center
        (0.22, 0.0),      # right
        (-0.22, 0.0),     # left
        (0.0, 0.22),      # top
        (0.0, -0.22),     # bottom
    ]
    hole_radius = 0.12  # radius in normalized coords
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Check if this pixel is inside any hole
            is_hole = False
            for hx, hy in hole_centers:
                dx = x - hx
                dy = y - hy
                if dx * dx + dy * dy < hole_radius * hole_radius:
                    is_hole = True
                    break
            
            # 1 = dielectric, 0 = air (hole)
            row.append(0 if is_hole else 1)
        cell.append(row)
    
    # Verify symmetry (should be satisfied by construction, but let's be safe)
    # The pattern is symmetric by construction since holes are placed symmetrically
    
    return cell
# EVOLVE-BLOCK-END
