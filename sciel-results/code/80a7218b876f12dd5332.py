import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    
    # Use a 3x3 grid of 9 squares plus a 10th square in the remaining strip.
    # This achieves a sum of side lengths of approximately 3.0.
    
    # To avoid any floating point issues with the overlap check,
    # we use a very small epsilon to ensure the 10th square has a positive side
    # and is strictly inside the unit square, while the 3x3 grid squares
    # are also strictly inside.
    
    # The 3x3 grid squares have side s.
    # They occupy [0, 3s] x [0, 3s].
    # The 10th square has side t and is placed in the strip [3s, 1] x [0, 1] or similar.
    # To maximize the sum, we want s to be as large as possible, and t to be as large as possible.
    # If we place the 10th square in the vertical strip to the right of the grid:
    # Its x-range is [3s, 3s+t] and y-range is [0, t] (or centered).
    # Constraint: 3s + t <= 1.
    # To maximize 9s + t, we set t = 1 - 3s.
    # Sum = 9s + 1 - 3s = 6s + 1.
    # This is maximized when s is as large as possible.
    # The constraint s > 0 is obvious. The constraint that the grid fits is 3s <= 1.
    # So s can be at most 1/3.
    # If s = 1/3, then t = 0, which is not allowed (side > 0).
    # So we set s slightly less than 1/3.
    # Let s = 1/3 - eps.
    # Then t = 1 - 3(1/3 - eps) = 3*eps.
    # Sum = 9(1/3 - eps) + 3*eps = 3 - 9*eps + 3*eps = 3 - 6*eps.
    
    # We need to ensure that the squares are strictly inside [-tol, 1+tol] and do not overlap.
    # The overlap condition is |cx_i - cx_j| < (s_i + s_j)/2 - tol.
    # For adjacent squares in the grid, the distance is s, and the threshold is (s+s)/2 - tol = s - tol.
    # Since s < s - tol is false (s = s - 0), we have |cx_i - cx_j| = s, which is NOT < s - tol.
    # Wait, the condition is |cx_i - cx_j| < t_overlap, where t_overlap = (s_i + s_j)/2 - tol.
    # If |cx_i - cx_j| = s, and t_overlap = s - tol, then s < s - tol is False. So they do NOT overlap.
    # This is correct because they are touching.
    
    # For the 10th square and the grid squares:
    # The 10th square is at x in [3s, 3s+t], y in [0, t] (let's say centered at (3s + t/2, t/2)).
    # The rightmost grid square is at x in [2s, 3s], y in [0, s] (for the bottom-right one).
    # Distance in x: (3s + t/2) - (2s + s/2) = 3s + t/2 - 2.5s = 0.5s + t/2.
    # Threshold: (s + t)/2 - tol.
    # We need 0.5s + t/2 >= (s + t)/2 - tol.
    # 0.5s + t/2 >= 0.5s + t/2 - tol.
    # This is always true. So they do not overlap.
    
    # Let's place the 10th square at the top of the strip to be safe.
    # x in [3s, 3s+t], y in [1-t, 1].
    # Center: (3s + t/2, 1 - t/2).
    # Distance to top-right grid square (center (2.5s, 2.5s)):
    # dx = (3s + t/2) - 2.5s = 0.5s + t/2.
    # dy = (1 - t/2) - 2.5s.
    # Since 3s + t = 1, t = 1 - 3s.
    # dy = 1 - (1-3s)/2 - 2.5s = 1 - 0.5 + 1.5s - 2.5s = 0.5 - s.
    # We need dx >= (s+t)/2 - tol AND dy >= (s+t)/2 - tol.
    # dx = 0.5s + (1-3s)/2 = 0.5s + 0.5 - 1.5s = 0.5 - s.
    # So dx = dy = 0.5 - s.
    # Threshold = (s + 1 - 3s)/2 - tol = (1 - 2s)/2 - tol = 0.5 - s - tol.
    # We need 0.5 - s >= 0.5 - s - tol. This is always true.
    
    # So the layout is valid for any s < 1/3.
    
    eps = 1e-7
    s = (1.0 - 3*eps) / 3.0
    t = 1.0 - 3*s
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    idx = 0
    for i in range(3):
        for j in range(3):
            cx = (j + 0.5) * s
            cy = (i + 0.5) * s
            centers[idx, 0] = cx
            centers[idx, 1] = cy
            sides[idx] = s
            idx += 1
    
    # 10th square in the top-right strip
    cx = 3*s + t/2
    cy = 1.0 - t/2
    centers[idx, 0] = cx
    centers[idx, 1] = cy
    sides[idx] = t
    
    return centers, sides
