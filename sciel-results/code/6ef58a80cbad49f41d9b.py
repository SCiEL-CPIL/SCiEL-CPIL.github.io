# EVOLVE-BLOCK-START
import numpy as np
import time

TOL = 1e-7


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    t0 = time.time()
    DEADLINE = t0 + 570.0  # safety margin
    n = 10

    def _is_feasible(centers, sides, tol=TOL):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i] / 2.0 < -tol or cx[i] + s[i] / 2.0 > 1.0 + tol:
                return False
            if cy[i] - s[i] / 2.0 < -tol or cy[i] + s[i] / 2.0 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def _score(centers, sides):
        if not _is_feasible(centers, sides):
            return -np.inf
        return np.sum(sides)

    # --- Analytical good starting layouts ---
    # Layout 1: 4 rows of 2+2+2+4 (or similar)
    # Let's think about what maximizes sum of sides.
    # If we have a grid of r x c squares, each side = 1/max(r,c) roughly.
    # Actually, if all squares same size s, and we pack them in a grid:
    # floor(1/s) * floor(1/s) >= 10
    # 3*3=9 < 10, so need at least one row with 4 or one col with 4.
    # 4x3 grid: 12 squares, side 0.25. Sum = 10*0.25 = 2.5. But we only need 10.
    # We can make the last row have 2 squares of larger size?
    # Or mix sizes.
    
    # Let's try a specific layout:
    # Row 0: 4 squares of side s1
    # Row 1: 3 squares of side s2
    # Row 2: 3 squares of side s3
    # Total 10 squares.
    # Constraints:
    # 4*s1 <= 1  => s1 <= 0.25
    # 3*s2 <= 1  => s2 <= 1/3
    # 3*s3 <= 1  => s3 <= 1/3
    # s1 + s2 + s3 <= 1
    # To maximize sum = 4*s1 + 3*s2 + 3*s3
    # Set s2 = s3 = 1/3, then s1 <= 1 - 2/3 = 1/3. But s1 <= 0.25.
    # So s1 = 0.25, s2 = s3 = 1/3.
    # Sum = 4*0.25 + 3*(1/3) + 3*(1/3) = 1 + 1 + 1 = 3.0.
    # Check feasibility:
    # Row 0: 4 squares of 0.25. Width = 1.0. Height = 0.25.
    # Row 1: 3 squares of 1/3. Width = 1.0. Height = 1/3.
    # Row 2: 3 squares of 1/3. Width = 1.0. Height = 1/3.
    # Total height = 0.25 + 1/3 + 1/3 = 0.25 + 2/3 = 0.9166... <= 1.0.
    # This is feasible! Sum = 3.0.

    # Can we do better?
    # What if Row 0 has 3 squares of 1/3, Row 1 has 3 squares of 1/3, Row 2 has 4 squares of 0.25?
    # Same sum.
    
    # What if we don't align rows perfectly?
    # Maybe 2 rows of 5 squares of 0.2? Sum = 2.0. Worse.
    # Maybe 1 row of 5 (0.2), 1 row of 5 (0.2)? Sum = 2.0.
    
    # What about 1 big square and 9 small?
    # 1 square of side 0.5. Remaining space: 0.5x1 strip on right, 1x0.5 strip on bottom? No, it's a square.
    # If square is at (0,0) to (0.5,0.5), remaining is L-shape.
    # Hard to pack 9 squares efficiently in L-shape.
    
    # Let's stick with the 3.0 solution and try to optimize it locally.
    # We can also try other configurations.
    # 2 rows:
    # Row 0: 5 squares of 0.2.
    # Row 1: 5 squares of 0.2.
    # Sum = 2.0.
    
    # 1 row: 10 squares of 0.1. Sum = 1.0.
    
    # So 3.0 seems like a strong candidate.
    # Let's construct it.
    
    def _make_layout_30():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        idx = 0
        # Row 0: 4 squares of 0.25
        s1 = 0.25
        for j in range(4):
            centers[idx, 0] = j * s1 + s1 / 2
            centers[idx, 1] = s1 / 2
            sides[idx] = s1
            idx += 1
        # Row 1: 3 squares of 1/3
        s2 = 1.0 / 3.0
        y1 = s1 + s2 / 2
        for j in range(3):
            centers[idx, 0] = j * s2 + s2 / 2
            centers[idx, 1] = y1
            sides[idx] = s2
            idx += 1
        # Row 2: 3 squares of 1/3
        s3 = 1.0 / 3.0
        y2 = s1 + s2 + s3 / 2
        for j in range(3):
            centers[idx, 0] = j * s3 + s3 / 2
            centers[idx, 1] = y2
            sides[idx] = s3
            idx += 1
        return centers, sides

    best_centers, best_sides = _make_layout_30()
    best_score = _score(best_centers, best_sides)

    # Try other layouts
    # Layout: 2 rows of 3, 2 rows of 2?
    # Row 0: 3 squares of 1/3
    # Row 1: 3 squares of 1/3
    # Row 2: 2 squares of 0.5
    # Row 3: 2 squares of 0.5
    # Height: 1/3 + 1/3 + 0.5 + 0.5 = 2/3 + 1 = 1.666 > 1. Infeasible.
    
    # Layout: 3 rows of 3, 1 row of 1?
    # 3 squares of 1/3 (row 0)
    # 3 squares of 1/3 (row 1)
    # 3 squares of 1/3 (row 2)
    # 1 square of ?
    # Height so far: 1.0. No room for row 3.
    # So 9 squares of 1/3 fill the unit square.
    # The 10th square must be smaller or we rearrange.
    # If we have 9 squares of 1/3, they fill the square.
    # So we must make some squares smaller to fit the 10th.
    # Our 3.0 layout does this by making the top row smaller (0.25 vs 1/3).
    # Sum = 4*0.25 + 6*(1/3) = 1 + 2 = 3.0.
    
    # Can we do better than 3.0?
    # Suppose we have rows with heights h1, h2, h3, h4.
    # And squares of size s_i in each row.
    # Sum of heights <= 1.
    # Sum of squares = sum(s_i).
    # In each row, sum of widths of squares <= 1.
    # If a row has k squares of size s, then k*s <= 1.
    # To maximize sum, we want large s.
    # Large s means fewer squares per row.
    # But we need to fit 10 squares total.
    # Let's try 4 rows.
    # Row 0: 2 squares of 0.5
    # Row 1: 2 squares of 0.5
    # Row 2: 3 squares of 1/3
    # Row 3: 3 squares of 1/3
    # Height: 0.5 + 0.5 + 1/3 + 1/3 = 1 + 2/3 > 1. Infeasible.
    
    # Row 0: 2 squares of 0.4
    # Row 1: 2 squares of 0.4
    # Row 2: 3 squares of 1/3
    # Row 3: 3 squares of 1/3
    # Height: 0.4 + 0.4 + 1/3 + 1/3 = 0.8 + 0.666 = 1.466 > 1.
    
    # Row 0: 2 squares of 0.3
    # Row 1: 2 squares of 0.3
    # Row 2: 3 squares of 1/3
    # Row 3: 3 squares of 1/3
    # Height: 0.3 + 0.3 + 0.666 = 1.266 > 1.
    
    # It seems 3 rows is better.
    # Let's try 3 rows with different counts.
    # 2, 3, 5?
    # Row 0: 2 squares of 0.5
    # Row 1: 3 squares of 1/3
    # Row 2: 5 squares of 0.2
    # Height: 0.5 + 1/3 + 0.2 = 0.5 + 0.333 + 0.2 = 1.033 > 1.
    
    # 2, 4, 4?
    # Row 0: 2 squares of 0.5
    # Row 1: 4 squares of 0.25
    # Row 2: 4 squares of 0.25
    # Height: 0.5 + 0.25 + 0.25 = 1.0.
    # Sum = 2*0.5 + 4*0.25 + 4*0.25 = 1 + 1 + 1 = 3.0.
    # Same sum.
    
    # 3, 3, 4?
    # Row 0: 3 squares of 1/3
    # Row 1: 3 squares of 1/3
    # Row 2: 4 squares of 0.25
    # Height: 1/3 + 1/3 + 0.25 = 0.9166.
    # Sum = 3*(1/3) + 3*(1/3) + 4*0.25 = 1 + 1 + 1 = 3.0.
    
    # It seems 3.0 is a local maximum for grid layouts.
    # Can we break the grid?
    # Maybe interlocking squares?
    # But axis-aligned squares don't interlock easily without overlapping.
    # The constraint is non-overlapping interiors.
    # So they can touch.
    
    # Let's try to optimize the 3.0 layout locally.
    # We can increase the size of the top row squares if we reduce the height of the bottom rows?
    # No, the bottom rows are limited by width.
    # If we make the bottom row squares smaller, we can make the top row squares larger?
    # But the top row is limited by width too (4 squares).
    # So s1 <= 0.25.
    # And s2, s3 <= 1/3.
    # And s1 + s2 + s3 <= 1.
    # To maximize 4*s1 + 3*s2 + 3*s3, we set s1=0.25, s2=s3=1/3.
    # This is the maximum for this configuration.
    
    # What if we use a different configuration?
    # 1, 4, 5?
    # Row 0: 1 square of 1.0? No, then no room for others.
    # Row 0: 1 square of s1.
    # Row 1: 4 squares of s2.
    # Row 2: 5 squares of s3.
    # s1 <= 1, 4*s2 <= 1, 5*s3 <= 1.
    # s1 + s2 + s3 <= 1.
    # Maximize s1 + 4*s2 + 5*s3.
    # Set s2 = 0.25, s3 = 0.2.
    # s1 <= 1 - 0.25 - 0.2 = 0.55.
    # Sum = 0.55 + 4*0.25 + 5*0.2 = 0.55 + 1 + 1 = 2.55. Worse.
    
    # So 3.0 seems robust.
    # Let's implement a local search to see if we can improve it.
    
    def _local_search(centers, sides, max_iters=5000):
        best_c = centers.copy()
        best_s = sides.copy()
        best_val = _score(best_c, best_s)
        if not np.isfinite(best_val):
            return best_c, best_s, best_val
        
        step = 0.01
        for it in range(max_iters):
            if time.time() > DEADLINE - 30:
                break
            improved = False
            for i in range(n):
                # Try adjusting side length
                for delta in [step, -step]:
                    trial_s = best_s.copy()
                    trial_s[i] += delta
                    if trial_s[i] < 1e-4 or trial_s[i] > 0.99:
                        continue
                    if _is_feasible(best_c, trial_s):
                        new_val = np.sum(trial_s)
                        if new_val > best_val + 1e-10:
                            best_s = trial_s
                            best_val = new_val
                            improved = True
                # Try adjusting cx
                for delta in [step, -step]:
                    trial_c = best_c.copy()
                    trial_c[i, 0] += delta
                    if trial_c[i, 0] < 0 or trial_c[i, 0] > 1:
                        continue
                    if _is_feasible(trial_c, best_s):
                        new_val = np.sum(best_s)
                        if new_val > best_val + 1e-10:
                            best_c = trial_c
                            best_val = new_val
                            improved = True
                # Try adjusting cy
                for delta in [step, -step]:
                    trial_c = best_c.copy()
                    trial_c[i, 1] += delta
                    if trial_c[i, 1] < 0 or trial_c[i, 1] > 1:
                        continue
                    if _is_feasible(trial_c, best_s):
                        new_val = np.sum(best_s)
                        if new_val > best_val + 1e-10:
                            best_c = trial_c
                            best_val = new_val
                            improved = True
            
            if not improved:
                step *= 0.85
                if step < 1e-6:
                    break
        
        return best_c, best_s, best_val

    best_centers, best_sides, best_score = _local_search(best_centers, best_sides)

    # Try a different starting layout: 2 rows of 5
    def _make_layout_25():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            col = i % 5
            row = i // 5
            s = 0.2
            centers[i, 0] = (col + 0.5) * s
            centers[i, 1] = (row + 0.5) * s
            sides[i] = s
        return centers, sides

    c2, s2 = _make_layout_25()
    sc2 = _score(c2, s2)
    if sc2 > best_score:
        best_centers, best_sides, best_score = _local_search(c2, s2)

    # Try a random restart
    np.random.seed(42)
    for _ in range(10):
        if time.time() > DEADLINE - 60:
            break
        # Random layout
        centers = np.random.rand(n, 2)
        sides = np.random.rand(n) * 0.2 + 0.1
        # Scale to fit
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
            centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
        
        # Try to fix overlaps by shrinking
        for _ in range(100):
            if _is_feasible(centers, sides):
                break
            # Find overlapping pair
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j]) / 2 - TOL
                    if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                        # Shrink both
                        sides[i] *= 0.95
                        sides[j] *= 0.95
                        # Re-center
                        centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
                        centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
                        centers[j, 0] = np.clip(centers[j, 0], sides[j]/2, 1 - sides[j]/2)
                        centers[j, 1] = np.clip(centers[j, 1], sides[j]/2, 1 - sides[j]/2)
        
        if _is_feasible(centers, sides):
            val = _score(centers, sides)
            if val > best_score:
                best_centers, best_sides, best_score = _local_search(centers, sides)

    # Final check
    if not _is_feasible(best_centers, best_sides):
        # Fallback to simple grid
        best_centers, best_sides = _make_layout_30()
        best_score = _score(best_centers, best_sides)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
