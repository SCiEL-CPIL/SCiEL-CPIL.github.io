# EVOLVE-BLOCK-START
import numpy as np
import time


def _valid(centers, sides, tol=1e-7):
    """Check validity of a packing configuration."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if not np.all(sides > 0):
        return False
    # Boundary check
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Non-overlap check
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square to maximize sum of side lengths.
    
    Uses a constructive layout: 3x3 grid of nearly-equal squares filling most of the
    unit square, plus one small square in the corner. The theoretical supremum for
    10 squares is 3.0 (area argument: sum of areas <= 1, but for equal sizes the
    sum of sides can approach 3 when squares are ~1/3 each). We get very close.
    """
    start_time = time.time()
    deadline = 630.0 - 10.0  # safety margin
    
    # Constructive layout: 3x3 grid + 1 tiny corner square
    # The 9 grid squares have side s, filling [0, 3s] x [0, 3s]
    # The 10th square has side S, placed in the top-right corner [1-S, 1] x [1-S, 1]
    # For no overlap: 3s + S >= 1 (they touch or overlap on the diagonal)
    # Actually we need: the grid's right edge = 3s, corner's left edge = 1-S
    # No overlap requires: 3s <= 1-S OR they just touch.
    # Touching: 3s = 1-S => S = 1-3s
    # Sum = 9s + S = 9s + 1 - 3s = 6s + 1
    # To maximize, maximize s. Max s is when 3s <= 1, so s <= 1/3.
    # At s = 1/3: S = 0, sum = 3. But S must be > 0.
    # So take s = 1/3 - epsilon, S = 3*epsilon.
    # Sum = 9*(1/3 - eps) + 3*eps = 3 - 9*eps + 3*eps = 3 - 6*eps.
    
    # Use a small epsilon to keep S > 0 and satisfy tolerance
    eps = 1e-5
    s = (1.0 - 3.0 * eps) / 3.0  # so that 3*s + 3*eps = 1, S = 3*eps
    S = 3.0 * eps  # side of the corner square
    
    # Actually let's be more careful:
    # Grid squares: 3x3 grid, each with side s
    # Centered at ((col+0.5)*s, (row+0.5)*s) for col,row in 0,1,2
    # They occupy [0, 3s] x [0, 3s]
    # Corner square: side S, centered at (1 - S/2, 1 - S/2)
    # Occupies [1-S, 1] x [1-S, 1]
    # For no overlap with grid: need 3s <= 1-S (x-direction) AND 3s <= 1-S (y-direction)
    # So 3s + S <= 1
    # Sum = 9s + S
    # Maximize 9s + S subject to 3s + S <= 1, s > 0, S > 0
    # From constraint: S <= 1 - 3s, so sum <= 9s + 1 - 3s = 6s + 1
    # Max s is 1/3 (giving S=0). So take s close to 1/3.
    
    s = 0.33333  # 3s = 0.99999
    S = 0.00001  # 3s + S = 1.0 exactly
    # Check: 3*0.33333 = 0.99999, 0.99999 + 0.00001 = 1.0. Good.
    # Sum = 9*0.33333 + 0.00001 = 2.99997 + 0.00001 = 2.99998
    
    # Verify no overlap more carefully with tolerance
    # Grid right edge: 3*s = 0.99999
    # Corner left edge: 1 - S = 0.99999
    # They touch exactly. With tol=1e-7, t = (s+S)/2 - tol
    # For a grid square adjacent to corner: dx = |cx_grid - cx_corner|
    # Rightmost grid col center: 2.5*s = 2.5*0.33333 = 0.833325
    # Corner center x: 1 - S/2 = 1 - 0.000005 = 0.999995
    # dx = 0.999995 - 0.833325 = 0.16667
    # t = (0.33333 + 0.00001)/2 - 1e-7 = 0.16667 - 1e-7 = 0.1666699
    # dx > t? 0.16667 > 0.1666699. Yes, just barely.
    # Actually let's compute more carefully:
    # dx = (1 - S/2) - 2.5*s = 1 - S/2 - 2.5s
    # t = (s + S)/2 - tol
    # Need dx >= t: 1 - S/2 - 2.5s >= (s+S)/2 - tol
    # 1 - S/2 - 2.5s >= s/2 + S/2 - tol
    # 1 - 3s - S >= -tol
    # 1 - 3*0.33333 - 0.00001 >= -1e-7
    # 1 - 0.99999 - 0.00001 = 0 >= -1e-7. True.
    
    # Build centers and sides
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    # 9 grid squares
    idx = 0
    for row in range(3):
        for col in range(3):
            centers[idx, 0] = (col + 0.5) * s
            centers[idx, 1] = (row + 0.5) * s
            sides[idx] = s
            idx += 1
    
    # 1 corner square
    centers[idx, 0] = 1.0 - S / 2
    centers[idx, 1] = 1.0 - S / 2
    sides[idx] = S
    
    # Validate
    if not _valid(centers, sides):
        # Fallback to simple grid
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
    
    return centers, sides


# EVOLVE-BLOCK-END
