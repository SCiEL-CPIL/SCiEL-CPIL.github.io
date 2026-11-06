"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a 32x32 symmetric unit cell using concentric square rings
    with alternating dielectric/air assignments to maximize the TM
    band gap between bands 4 and 5.
    
    The approach uses four-fold symmetry (D4) to reduce the design space.
    We parameterize the cell by concentric square rings, where each ring
    is either dielectric (1) or air (0). This creates periodic microstructures
    beyond simple rods that can open larger higher-order band gaps.
    """
    size = problem["pixels"]
    half = size // 2  # 16
    
    # For a 32x32 cell with 4-fold symmetry about the center,
    # we can define the top-left 16x16 quadrant and mirror it.
    # Concentric square rings: ring k consists of pixels where max(|i-15.5|, |j-15.5|) is in [k-0.5, k+0.5)
    # Actually, let's think in terms of Chebyshev distance from center.
    
    # Center is at (15.5, 15.5) in pixel coordinates (0-indexed).
    # For pixel (i, j), the Chebyshev distance from center is max(|i - 15.5|, |j - 15.5|)
    # Ring index r = floor(max(|i - 15.5|, |j - 15.5|) + 0.5) = floor(max(i, j, size-1-i, size-1-j)/1) ... 
    # Let me reconsider. For pixel (i,j) in 0..31:
    # distance from center along i: |i - 15.5|
    # distance from center along j: |j - 15.5|
    # Chebyshev distance: d = max(|i - 15.5|, |j - 15.5|)
    # Ring index: r = int(d + 0.5) which gives 0..15
    
    # So we have 16 rings, each assigned 0 or 1.
    # We want to find a good binary sequence of length 16.
    
    # Let's try a few candidate patterns:
    # Pattern 1: alternating starting with dielectric at center
    # Pattern 2: alternating starting with air at center
    # Pattern 3: other patterns
    
    # For now, let's use a heuristic: try several patterns and pick one.
    # Since we can't evaluate the band structure here, we'll use a known good pattern
    # for higher-order TM gaps.
    
    # A common design for higher-order gaps uses a checkerboard-like pattern
    # or specific ring assignments. Let's try:
    # Rings from center outward: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    # This gives a checkerboard-like structure at the pixel level.
    
    # Actually, let me think about this more carefully. The seed uses a circular rod.
    # For higher-order gaps, we need more complex structures.
    
    # Let's try a pattern where we have dielectric in rings 0,2,4,6,8,10,12,14
    # and air in the others. This creates a square lattice of dielectric squares.
    
    ring_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Compute Chebyshev distance from center
            di = abs(i - (size - 1) / 2.0)
            dj = abs(j - (size - 1) / 2.0)
            d = max(di, dj)
            ring = int(d + 0.5)
            # Clamp to valid range
            ring = min(ring, 15)
            row.append(ring_pattern[ring])
        cell.append(row)
    
    return cell
# EVOLVE-BLOCK-END
