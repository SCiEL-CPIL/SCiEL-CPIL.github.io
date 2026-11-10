# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

    def _build_grid_layout(cols_per_row, side_per_col, y_positions):
        """Build centers and sides from grid specification."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        idx = 0
        for r_idx, row_cols in enumerate(cols_per_row):
            if idx >= n:
                break
            y = y_positions[r_idx]
            x_current = 0.0
            for c in row_cols:
                if idx >= n:
                    break
                s = side_per_col[c]
                centers[idx, 0] = x_current + s / 2.0
                centers[idx, 1] = y
                sides[idx] = s
                x_current += s
                idx += 1
        return centers, sides

    def _check_grid_feasible(cols_per_row, side_per_col, y_positions, tol=TOL):
        """Check if grid layout is feasible."""
        centers, sides = _build_grid_layout(cols_per_row, side_per_col, y_positions)
        return _is_feasible(centers, sides, tol)

    def _grid_objective(cols_per_row, sides, tol=TOL):
        """
        Given column counts per row and side lengths per column,
        compute the sum of sides if feasible, else -inf.
        y_positions are computed as cumulative sum of max side in each row.
        """
        r = len(cols_per_row)
        y_positions = np.zeros(r)
        y_bottom = 0.0
        for r_idx in range(r):
            row_cols = cols_per_row[r_idx]
            # Height of this row is the max side length of squares in this row
            row_height = max(sides[c] for c in row_cols)
            y_positions[r_idx] = y_bottom + row_height / 2.0
            y_bottom += row_height
            if y_bottom > 1.0 + tol:
                return -np.inf
            # Check width
            row_width = sum(sides[c] for c in row_cols)
            if row_width > 1.0 + tol:
                return -np.inf
        
        # Build and check feasibility
        centers, sides_arr = _build_grid_layout(cols_per_row, sides, y_positions)
        if not _is_feasible(centers, sides_arr, tol):
            return -np.inf
        
        return np.sum(sides_arr)

    def _ternary_search_1d(f, a, b, tol=1e-8, max_iter=100):
        """Ternary search for maximum of f on [a, b]."""
        for _ in range(max_iter):
            if b - a < tol:
                break
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            if f(m1) < f(m2):
                a = m1
            else:
                b = m2
        return (a + b) / 2.0

    def _optimize_grid(cols_per_row, n_cols):
        """
        Optimize side lengths for a given grid structure.
        cols_per_row: list of lists, each inner list contains column indices for that row.
        n_cols: total number of distinct column sizes.
        """
        r = len(cols_per_row)
        
        def obj(sides_vec):
            return _grid_objective(cols_per_row, sides_vec, TOL)
        
        # Initial guess: equal sides
        s0 = np.ones(n_cols) * (1.0 / max(len(cols_per_row[0]), 1))
        
        # Try multiple initial points
        best_val = -np.inf
        best_s = s0.copy()
        
        # Initial point 1: uniform
        val = obj(s0)
        if val > best_val:
            best_val = val
            best_s = s0.copy()
        
        # Initial point 2: based on row constraints
        s_init2 = np.zeros(n_cols)
        for c_idx in range(n_cols):
            # Find rows that contain this column
            min_side = 1.0
            for r_idx in range(r):
                if c_idx in cols_per_row[r_idx]:
                    row_cols = cols_per_row[r_idx]
                    # If all columns in row have same side, max side is 1/len(row_cols)
                    # But sides can differ, so this is just a heuristic
                    pass
            s_init2[c_idx] = 1.0 / max(1, n_cols)
        
        val = obj(s_init2)
        if val > best_val:
            best_val = val
            best_s = s_init2.copy()
        
        # Coordinate descent on side lengths
        step = 0.05
        for iter_cd in range(50):
            if time.time() > DEADLINE - 60:
                break
            improved = False
            for c_idx in range(n_cols):
                # Try increasing
                trial_s = best_s.copy()
                trial_s[c_idx] += step
                val = obj(trial_s)
                if val > best_val + 1e-10:
                    best_s = trial_s
                    best_val = val
                    improved = True
                # Try decreasing
                trial_s = best_s.copy()
                trial_s[c_idx] -= step
                if trial_s[c_idx] > 1e-6:
                    val = obj(trial_s)
                    if val > best_val + 1e-10:
                        best_s = trial_s
                        best_val = val
                        improved = True
            if not improved:
                step *= 0.7
                if step < 1e-6:
                    break
        
        return best_s, best_val

    def _enumerate_column_distributions(num_rows):
        """
        Enumerate all ways to distribute 10 squares into num_rows rows,
        where each row has at least 1 square, and squares are assigned to columns.
        Returns list of (cols_per_row, n_cols) where cols_per_row[r] is list of column indices.
        """
        results = []
        
        def _recurse(row_idx, remaining, current_cols_per_row, max_col_idx):
            if row_idx == num_rows:
                if remaining == 0:
                    results.append((current_cols_per_row[:], max_col_idx + 1))
                return
            
            # Try all possible numbers of squares in this row: 1 to remaining
            for num_in_row in range(1, remaining + 1):
                # The squares in this row will be assigned to consecutive columns
                # starting from max_col_idx
                cols_in_row = list(range(max_col_idx, max_col_idx + num_in_row))
                current_cols_per_row.append(cols_in_row)
                _recurse(row_idx + 1, remaining - num_in_row, current_cols_per_row, max_col_idx + num_in_row)
                current_cols_per_row.pop()
        
        _recurse(0, 10, [], 0)
        return results

    # Enumerate all grid configurations
    best_centers = None
    best_sides = None
    best_score = 0.0
    
    # Try 1 to 5 rows
    for num_rows in range(1, 6):
        if time.time() > DEADLINE - 120:
            break
        
        configs = _enumerate_column_distributions(num_rows)
        
        for cols_per_row, n_cols in configs:
            if time.time() > DEADLINE - 60:
                break
            
            # Skip if any row has too many columns (width constraint)
            # Max side is 1, so a row with k columns needs at least k*min_side <= 1
            # This is handled in the feasibility check
            
            s_opt, val = _optimize_grid(cols_per_row, n_cols)
            
            if val > best_score and val > 0:
                # Rebuild the layout
                r = len(cols_per_row)
                y_positions = np.zeros(r)
                y_bottom = 0.0
                for r_idx in range(r):
                    y_positions[r_idx] = y_bottom + s_opt[r_idx] / 2.0
                    y_bottom += s_opt[r_idx]
                
                centers, sides = _build_grid_layout(cols_per_row, s_opt, y_positions)
                
                if _is_feasible(centers, sides):
                    best_score = val
                    best_centers = centers
                    best_sides = sides
    
    # If no grid solution found, use a simple fallback
    if best_centers is None:
        # 2 rows of 5 squares each, side 0.2
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
    def _polish(centers, sides, max_iters=3000):
        best_c = centers.copy()
        best_s = sides.copy()
        best_score_local = np.sum(sides)
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
                        new_score = np.sum(trial_s)
                        if new_score > best_score_local + 1e-10:
                            best_s = trial_s
                            best_score_local = new_score
                            improved = True
                # Try adjusting cx
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
                # Try adjusting cy
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
    
    # Final check
    if not _is_feasible(best_centers, best_sides):
        # Fallback to simple grid
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
