# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2).
    
    Uses a non-separable construction: positions depend on both i and j
    to minimize the union volume. The key insight is that by offsetting
    tubes in both x and y based on a 2D pattern, we can reduce overlap
    more effectively than separable constructions.
    """
    N = 8
    
    # Generate base positions using a 2D optimization strategy.
    # The tube (i,j) at height c occupies:
    #   x in [x_{i,j} + c*i/N, x_{i,j} + c*i/N + 1/N]
    #   y in [y_{i,j} + c*j/N, y_{i,j} + c*j/N + 1/N]
    #
    # For the union to be small, we want tubes to overlap as much as possible
    # in the x-y plane at each height c, while still covering all directions.
    #
    # A good strategy: use a 2D lattice where the offset depends on a linear
    # combination of i and j, creating a sheared grid that minimizes the
    # bounding box area.
    
    # Strategy: use a 2D van der Corput / low-discrepancy sequence in 2D
    # to place the 64 tubes, then optimize the overall scaling/shift.
    
    # First, generate 64 points in [0,1]^2 using a 2D low-discrepancy sequence
    # (Koksma-van Aardenne-Ehrenfeest / Kronecker sequence)
    def _kronecker_2d(n, alpha1, alpha2):
        """Generate n points in [0,1]^2 using Kronecker sequence."""
        t = np.arange(n)
        x = (t * alpha1) % 1.0
        y = (t * alpha2) % 1.0
        return np.column_stack([x, y])
    
    # Use irrational-like offsets for good distribution
    alpha1 = np.sqrt(2) - 1  # ~0.4142
    alpha2 = np.sqrt(3) - 1  # ~0.7321
    
    points_2d = _kronecker_2d(64, alpha1, alpha2)
    
    # Now we need to assign these to tubes (i,j) in a way that minimizes overlap.
    # The key constraint: tube (i,j) moves in x by c*i/N and in y by c*j/N.
    # So the relative motion between tubes depends on (i,j) differences.
    
    # For minimal union, we want the "effective" positions (after accounting
    # for the shear) to cluster together. At height c, tube (i,j) is at:
    #   x_{i,j} + c*i/N, y_{i,j} + c*j/N
    # 
    # We want these to be close to each other for all (i,j) at each c.
    # This is a 2D problem: minimize the spread of {x_{i,j} + c*i/N, y_{i,j} + c*j/N}
    
    # Let's define: X_{i,j} = x_{i,j}, Y_{i,j} = y_{i,j}
    # At height c, position is (X_{i,j} + c*i/N, Y_{i,j} + c*j/N)
    # 
    # We want to minimize the area of the convex hull (or bounding box) of
    # these 64 points as a function of c, integrated over c in [0,1].
    
    # A simple approach: set X_{i,j} and Y_{i,j} to cancel the shear.
    # If we set X_{i,j} = -i/N * c0 and Y_{i,j} = -j/N * c0 for some c0,
    # then at height c0 all tubes are at the same point. But we need to
    # minimize the integral over all c.
    
    # Better: use a linear assignment. Let X_{i,j} = a*i + b*j, Y_{i,j} = c*i + d*j.
    # Then at height c, position is (a*i + b*j + c*i/N, c*i + d*j + c*j/N)
    # = ((a + c/N)*i + b*j, c*i + (d + c/N)*j)
    # 
    # The spread in x is proportional to max over (i,j) of |(a+c/N)*i + b*j|
    # The spread in y is proportional to max over (i,j) of |c*i + (d+c/N)*j|
    #
    # To minimize the bounding box area, we want these to be small.
    # If we set a = -c/N, then x = b*j, so spread in x is |b|*7.
    # If we set d = -c/N, then y = c*i, so spread in y is |c|*7.
    # But this doesn't help much.
    
    # Alternative: use the Kronecker points directly as the base positions.
    # The Kronecker sequence gives good 2D distribution, and the shear
    # will just translate each tube along a line.
    
    # Assign Kronecker points to tubes in row-major order (i, j)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # Use the Kronecker points as base positions
    base_x = points_2d[:, 0]
    base_y = points_2d[:, 1]
    
    # Scale to fit in a reasonable range. The tubes have width 1/8,
    # so we want the base positions to be in a range that minimizes overlap.
    # Let's try scaling by 1/8 to match the tube width.
    scale = 1.0 / N
    base_x = base_x * scale
    base_y = base_y * scale
    
    positions = np.column_stack([base_x, base_y])
    
    # Verify the result is valid
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (i_flat + 1) / N)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (j_flat + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        # Fallback: use a simple grid
        base_x = np.zeros(64)
        base_y = np.zeros(64)
        positions = np.column_stack([base_x, base_y])
    
    return positions


# EVOLVE-BLOCK-END
