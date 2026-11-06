"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with four-fold symmetry.
    Uses concentric square rings with alternating dielectric/air assignments.
    The pattern is parameterized by a binary sequence of ring thicknesses.
    
    Strategy: Explore patterns that maximize the TM band gap between bands 4 and 5.
    We use a heuristic based on known high-gap structures and symmetry.
    
    The unit cell is divided into concentric square rings. Each ring is either
    all dielectric (1) or all air (0). Due to 4-fold symmetry, we only need
    to specify the pattern for the first quadrant, but we enforce full symmetry
    (180-degree rotation and reflection) as required.
    
    For a 32x32 cell, the center is between pixels. We define rings by their
    Chebyshev distance from the center.
    """
    size = problem["pixels"]
    half = size // 2
    
    # Create a base pattern using concentric square rings
    # The pattern is defined by a binary sequence for each ring
    # Ring 0 is the innermost (center), ring 15 is the outermost
    
    # We'll use a pattern that alternates but with some optimization
    # Known good patterns for TM gaps often have specific ring configurations
    
    # Let's try a pattern based on maximizing dielectric contrast at certain radii
    # The pattern is: ring index -> 0 or 1
    
    # For 32x32, there are 16 possible ring distances (0 to 15)
    # We'll use a heuristic pattern
    
    # Pattern: start with dielectric at center, then alternate with some variation
    # Try: 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0 (alternating)
    # Or: 1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1 (period 4)
    # Or: 1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1 (period 4, shifted)
    
    # Let's use a pattern that has been found to work well in literature
    # for TM band gaps in square photonic crystals
    
    # Try the pattern: dielectric at even rings, air at odd rings
    # But with the outermost ring as air to reduce boundary effects
    
    ring_pattern = []
    for r in range(half):
        # Alternate: even rings are dielectric, odd are air
        # But we can tune this
        if r % 2 == 0:
            ring_pattern.append(1)
        else:
            ring_pattern.append(0)
    
    # Ensure the outermost ring is air for better confinement
    ring_pattern[-1] = 0
    
    # Create the cell
    cell = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            # Compute Chebyshev distance from center
            # Center is at (half - 0.5, half - 0.5) in pixel coordinates
            # Distance from center for pixel (i, j):
            dx = abs(i - (half - 0.5))
            dy = abs(j - (half - 0.5))
            cheb_dist = max(dx, dy)
            ring_index = int(cheb_dist)
            
            if ring_index < half:
                cell[i][j] = ring_pattern[ring_index]
    
    # Enforce symmetry: the cell should be invariant under:
    # 1. 180-degree rotation: cell[i][j] == cell[size-1-i][size-1-j]
    # 2. Reflection about vertical axis: cell[i][j] == cell[i][size-1-j]
    # 3. Reflection about horizontal axis: cell[i][j] == cell[size-1-i][j]
    # 4. Transpose: cell[i][j] == cell[j][i]
    
    # Our construction using Chebyshev distance already ensures 4-fold symmetry
    # Let's verify and fix if needed
    
    # Verify 180-degree rotation symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size-1-i][size-1-j]:
                # This shouldn't happen with our construction
                pass
    
    # Verify reflection symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[i][size-1-j]:
                pass
    
    # Verify transpose symmetry
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[j][i]:
                pass
    
    return cell
# EVOLVE-BLOCK-END
