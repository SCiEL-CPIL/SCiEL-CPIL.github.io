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
    DEADLINE = t0 + 580.0
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

    def _build_layout(cols_per_row, col_sides, y_offsets):
        """
        Build centers and sides array from grid specification.
        cols_per_row[r] = list of column indices for row r
        col_sides[c] = side length for column c
        y_offsets[r] = y-center for row r
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        idx = 0
        for r in range(len(cols_per_row)):
            for c in cols_per_row[r]:
                if idx >= n:
                    break
                side = col_sides[c]
                # x position: sum of sides of previous columns in this row + side/2
                x = 0.0
                for c_prev in cols_per_row[r]:
                    if c_prev < c:
                        x += col_sides[c_prev]
                    elif c_prev == c:
                        x += side / 2.0
                        break
                centers[idx, 0] = x
                centers[idx, 1] = y_offsets[r]
                sides[idx] = side
                idx += 1
        return centers, sides

    def _grid_score(cols_per_row, col_sides, tol=TOL):
        """
        Compute score for a grid layout. Returns -inf if infeasible.
        col_sides is indexed by column index.
        """
        r = len(cols_per_row)
        
        # Compute y positions: rows are stacked, each row has height = max side in that row
        # Actually, for a grid, all squares in a row should have the same height? No,
        # but if they're in the same row, their y-centers are the same, so their heights
        # must be the same for them to align. Wait, no - they can have different sizes
        # but then they wouldn't be a proper grid.
        
        # For a proper grid packing, let's assume all squares in the same row have
        # the same side length, and all squares in the same column have the same side length.
        # This is a restriction, but it simplifies the problem.
        
        # Actually, let me just check feasibility directly.
        y_positions = np.zeros(r)
        y_bottom = 0.0
        for r_idx in range(r):
            # The height of row r_idx is the max side length in that row
            row_max_side = max(col_sides[c] for c in cols_per_row[r_idx])
            y_positions[r_idx] = y_bottom + row_max_side / 2.0
            y_bottom += row_max_side
        
        if y_bottom > 1.0 + tol:
            return -np.inf
        
        # Check each row width
        for r_idx in range(r):
            row_width = sum(col_sides[c] for c in cols_per_row[r_idx])
            if row_width > 1.0 + tol:
                return -np.inf
        
        centers, sides = _build_layout(cols_per_row, col_sides, y_positions)
        if not _is_feasible(centers, sides, tol):
            return -np.inf
        
        return np.sum(sides)

    def _optimize_col_sides(cols_per_row, n_cols, max_iter=200):
        """
        Optimize column side lengths for a given grid structure.
        """
        # Initialize with equal sides
        col_sides = np.ones(n_cols) * 0.1
        
        # Coordinate descent
        step = 0.05
        best_val = _grid_score(cols_per_row, col_sides)
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 60:
                break
            improved = False
            for c in range(n_cols):
                # Try increasing
                trial = col_sides.copy()
                trial[c] += step
                val = _grid_score(cols_per_row, trial)
                if val > best_val + 1e-10:
                    col_sides = trial
                    best_val = val
                    improved = True
                # Try decreasing
                trial = col_sides.copy()
                trial[c] -= step
                if trial[c] > 1e-6:
                    val = _grid_score(cols_per_row, trial)
                    if val > best_val + 1e-10:
                        col_sides = trial
                        best_val = val
                        improved = True
            if not improved:
                step *= 0.7
                if step < 1e-6:
                    break
        
        return col_sides, best_val

    def _enumerate_grids(num_rows):
        """
        Enumerate all ways to partition 10 items into num_rows non-empty groups,
        where each group represents a row and the items in each row are consecutive
        column indices.
        """
        results = []
        
        def _recurse(row_idx, remaining, current, next_col):
            if row_idx == num_rows:
                if remaining == 0:
                    results.append((current[:], next_col))
                return
            
            for count in range(1, remaining + 1):
                cols_in_row = list(range(next_col, next_col + count))
                current.append(cols_in_row)
                _recurse(row_idx + 1, remaining - count, current, next_col + count)
                current.pop()
        
        _recurse(0, 10, [], 0)
        return results

    best_centers = None
    best_sides = None
    best_score = 0.0

    # Try different numbers of rows
    for num_rows in range(1, 6):
        if time.time() > DEADLINE - 120:
            break
        
        grids = _enumerate_grids(num_rows)
        
        for cols_per_row, n_cols in grids:
            if time.time() > DEADLINE - 60:
                break
            
            col_sides, val = _optimize_col_sides(cols_per_row, n_cols)
            
            if val > best_score and val > 0:
                # Rebuild layout
                r = len(cols_per_row)
                y_positions = np.zeros(r)
                y_bottom = 0.0
                for r_idx in range(r):
                    row_max_side = max(col_sides[c] for c in cols_per_row[r_idx])
                    y_positions[r_idx] = y_bottom + row_max_side / 2.0
                    y_bottom += row_max_side
                
                centers, sides = _build_layout(cols_per_row, col_sides, y_positions)
                
                if _is_feasible(centers, sides):
                    best_score = val
                    best_centers = centers
                    best_sides = sides

    # Fallback if no grid solution
    if best_centers is None:
        best_centers = np.zeros((n, 2))
        best_sides = np.zeros(n)
        for i in range(n):
            col = i % 5
            row = i // 5
            side = 0.2
            best_centers[i, 0] = (col + 0.5) * side
            best_centers[i, 1] = (row + 0.5) * side
            best_sides[i] = side
        best_score = np.sum(best_sides)

    # Final polish with coordinate descent
    def _polish(centers, sides, max_iters=2000):
        best_c = centers.copy()
        best_s = sides.copy()
        best_score_local = np.sum(sides)
        step = 0.01
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 30:
                break
            improved = False
            for i in range(n):
                for delta in [step, -step]:
                    trial_s = best_s.copy()
                    trial_s[i] += delta
                    if trial_s[i] < 1e-4 or trial_s[i] > 0.99:
                        continue
                    if _is_feasible(best_c, trial_s):
                        new_score = np.sum(trial_s)
                        if new_score > best_score_local + 1e-10:
                            best_s = trial_s
                            best_score_local = new_score
                            improved = True
                for delta in [step, -step]:
                    trial_c = best_c.copy()
                    trial_c[i, 0] += delta
                    if trial_c[i, 0] < 0 or trial_c[i, 0] > 1:
                        continue
                    if _is_feasible(trial_c, best_s):
                        new_score = np.sum(best_s)
                        if new_score > best_score_local + 1e-10:
                            best_c = trial_c
                            best_score_local = new_score
                            improved = True
                for delta in [step, -step]:
                    trial_c = best_c.copy()
                    trial_c[i, 1] += delta
                    if trial_c[i, 1] < 0 or trial_c[i, 1] > 1:
                        continue
                    if _is_feasible(trial_c, best_s):
                        new_score = np.sum(best_s)
                        if new_score > best_score_local + 1e-10:
                            best_c = trial_c
                            best_score_local = new_score
                            improved = True
            if not improved:
                step *= 0.85
                if step < 1e-6:
                    break
        
        return best_c, best_s, best_score_local

    best_centers, best_sides, best_score = _polish(best_centers, best_sides)

    if not _is_feasible(best_centers, best_sides):
        best_centers = np.zeros((n, 2))
        best_sides = np.zeros(n)
        for i in range(n):
            col = i % 5
            row = i // 5
            side = 0.2
            best_centers[i, 0] = (col + 0.5) * side
            best_centers[i, 1] = (row + 0.5) * side
            best_sides[i] = side

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
