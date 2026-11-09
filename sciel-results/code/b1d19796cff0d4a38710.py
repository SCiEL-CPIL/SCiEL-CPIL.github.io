import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    
    # New approach: 2x2 grid of large squares in the bottom-left,
    # plus a vertical strip of squares on the right side.
    # 
    # Layout:
    #   - 4 squares of side s in a 2x2 grid at bottom-left
    #   - 6 squares stacked vertically in the right strip
    #
    # Constraints:
    #   - 2s + 6*t <= 1  (horizontal: 2 large squares + 6 strip squares side by side? No, strip is vertical)
    #   - Actually, the right strip has width t and height 1.
    #   - The 2x2 grid occupies [0, 2s] x [0, 2s]
    #   - The right strip occupies [2s, 2s+t] x [0, 1]
    #   - So 2s + t <= 1
    #   - The 6 squares in the strip each have side t, stacked vertically: 6t <= 1
    #   - So t <= 1/6
    #   - To maximize 4s + 6t subject to 2s + t <= 1 and t <= 1/6
    #
    # Let's try t = 1/6, then 2s <= 1 - 1/6 = 5/6, so s <= 5/12
    # Sum = 4*(5/12) + 6*(1/6) = 20/12 + 1 = 5/3 + 1 = 8/3 ≈ 2.667
    #
    # Alternatively, try to balance: 2s + t = 1, 6t = 1 => t = 1/6, s = 5/12
    # Or maybe different arrangement.
    #
    # Let me try a different topology: 
    # 5 squares of side 1/5 in a row? No, that's only 5.
    #
    # Let's try: 3 squares in top row, 3 in middle row, 3 in bottom row, 1 extra
    # Similar to current but different arrangement.
    #
    # Actually, let me try a simple 2x5 grid of squares with side 1/5 = 0.2
    # 10 squares of side 0.2, sum = 2.0. Too low.
    #
    # Let me try: 4 squares of side 0.25 in a 2x2 grid (occupying [0,0.5]x[0,0.5]),
    # and 6 squares of side 0.25 in the remaining space?
    # Remaining space: [0.5,1]x[0,1] and [0,0.5]x[0.5,1]
    # In [0.5,1]x[0,0.5]: can fit 2 squares of side 0.25
    # In [0.5,1]x[0.5,1]: can fit 2 squares of side 0.25
    # In [0,0.5]x[0.5,1]: can fit 2 squares of side 0.25
    # Total: 4 + 2 + 2 + 2 = 10 squares of side 0.25, sum = 2.5
    #
    # Better: use the current 3x3 grid approach but optimize.
    # 9 squares of side s in 3x3 grid: 3s <= 1, so s <= 1/3
    # 10th square in the leftover space.
    # Leftover is a thin strip of width 1-3s.
    # If s = 1/3 - eps, strip width = 3*eps.
    # 10th square has side 3*eps.
    # Sum = 9*(1/3 - eps) + 3*eps = 3 - 9*eps + 3*eps = 3 - 6*eps
    #
    # This is what the current program does approximately.
    #
    # Let me try a completely different approach:
    # 2 rows of 5 squares each.
    # Each square has side s.
    # 5s <= 1, so s <= 0.2
    # 2s <= 1, so s <= 0.5
    # So s = 0.2, sum = 10*0.2 = 2.0
    #
    # Or 4 rows: 3,3,3,1 squares
    # 3s <= 1, s <= 1/3
    # 4s <= 1, s <= 1/4
    # So s = 1/4, sum = 9*0.25 + 0.25 = 2.5
    #
    # The 3x3+strip approach with sum ~3.0 seems best.
    # Let me just implement it cleanly.
    
    eps = 1e-7
    s = (1.0 - 3*eps) / 3.0  # Side of 9 squares
    
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
    
    # 10th square in the strip
    strip_width = 1.0 - 3*s  # = 3*eps
    cx = 3*s + strip_width/2
    cy = 0.5
    centers[idx, 0] = cx
    centers[idx, 1] = cy
    sides[idx] = strip_width
    
    return centers, sides
