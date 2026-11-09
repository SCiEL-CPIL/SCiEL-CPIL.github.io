# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 625  # leave 5s margin before 630s kill

    def is_feasible(c, s):
        """Check if configuration (centers, sides) is feasible."""
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si / 2 < -tol or cx + si / 2 > 1 + tol:
                return False
            if cy - si / 2 < -tol or cy + si / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                if abs(c[i][0] - c[j][0]) < t and abs(c[i][1] - c[j][1]) < t:
                    return False
        return True

    def objective(c, s):
        return np.sum(s)

    def guillotine_packing(rect, squares_idx, depth=0):
        """
        Recursively pack squares into a rectangle using guillotine cuts.
        rect: (x, y, w, h) - bottom-left corner and width/height
        squares_idx: list of square indices to place in this region
        Returns: (centers, sides) arrays or None if infeasible
        """
        if len(squares_idx) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        
        if len(squares_idx) == 1:
            i = squares_idx[0]
            x, y, w, h = rect
            # Place square centered in the rectangle
            cx = x + w / 2
            cy = y + h / 2
            side = min(w, h)
            return np.array([[cx, cy]]), np.array([side])
        
        # Try all ways to split the set into two non-empty subsets
        # and all cut positions (horizontal and vertical)
        best_result = None
        best_score = -1
        
        # For small sets, try all partitions
        if len(squares_idx) <= 6:
            # Generate all subsets of size 1 to len-1
            from itertools import combinations
            for k in range(1, len(squares_idx)):
                for subset in combinations(squares_idx, k):
                    subset = list(subset)
                    complement = [i for i in squares_idx if i not in subset]
                    
                    # Try vertical cut
                    # The cut position determines the width available to each side
                    # We need to find a split ratio that works
                    x, y, w, h = rect
                    
                    # For vertical cut at position fx (fraction of width):
                    # Left rect: (x, y, fx*w, h), Right rect: (x+fx*w, y, (1-fx)*w, h)
                    # We need both sub-packs to be feasible
                    # Binary search on fx won't work directly; try several values
                    for fx in [0.25, 0.33, 0.4, 0.45, 0.5, 0.55, 0.6, 0.67, 0.75]:
                        left_rect = (x, y, fx * w, h)
                        right_rect = (x + fx * w, y, (1 - fx) * w, h)
                        
                        result_left = guillotine_packing(left_rect, subset, depth + 1)
                        if result_left is None:
                            continue
                        result_right = guillotine_packing(right_rect, complement, depth + 1)
                        if result_right is None:
                            continue
                        
                        c_left, s_left = result_left
                        c_right, s_right = result_right
                        c_all = np.vstack([c_left, c_right])
                        s_all = np.concatenate([s_left, s_right])
                        score = np.sum(s_all)
                        
                        if score > best_score:
                            best_score = score
                            best_result = (c_all, s_all)
                    
                    # Try horizontal cut
                    for fy in [0.25, 0.33, 0.4, 0.45, 0.5, 0.55, 0.6, 0.67, 0.75]:
                        bottom_rect = (x, y, w, fy * h)
                        top_rect = (x, y + fy * h, w, (1 - fy) * h)
                        
                        result_bottom = guillotine_packing(bottom_rect, subset, depth + 1)
                        if result_bottom is None:
                            continue
                        result_top = guillotine_packing(top_rect, complement, depth + 1)
                        if result_top is None:
                            continue
                        
                        c_bottom, s_bottom = result_bottom
                        c_top, s_top = result_top
                        c_all = np.vstack([c_bottom, c_top])
                        s_all = np.concatenate([s_bottom, s_top])
                        score = np.sum(s_all)
                        
                        if score > best_score:
                            best_score = score
                            best_result = (c_all, s_all)
        
        if best_result is None:
            return None
        
        return best_result

    def guillotine_optimize():
        """
        Use guillotine cut tree to find a good packing.
        We'll try different orderings of squares and optimize the tree structure.
        """
        best_c = None
        best_s = None
        best_val = -1
        
        # Strategy: try different square size orderings and tree structures
        # Start with equal-sized squares and grow
        
        # Method 1: Recursive splitting with optimized ratios
        # We'll use a parameterized approach where we search over split ratios
        
        # First, try a fixed good structure
        # 2x5 grid: split vertically into 5 columns, each split horizontally into 2 rows
        # Or 5x2 grid
        # Or 1 big + 9 small, etc.
        
        # Let's try several template structures
        
        # Template A: 2 columns x 5 rows
        # Split unit square vertically at 0.5, then each column horizontally into 5
        x, y, w, h = 0, 0, 1, 1
        col_width = w / 2
        row_height = h / 5
        
        centers = []
        sides = []
        for col in range(2):
            for row in range(5):
                cx = x + col * col_width + col_width / 2
                cy = y + row * row_height + row_height / 2
                side = min(col_width, row_height)
                centers.append([cx, cy])
                sides.append(side)
        
        c_a = np.array(centers)
        s_a = np.array(sides)
        if is_feasible(c_a, s_a):
            val = objective(c_a, s_a)
            if val > best_val:
                best_val = val
                best_c = c_a.copy()
                best_s = s_a.copy()
        
        # Template B: 5 columns x 2 rows
        col_width = w / 5
        row_height = h / 2
        centers = []
        sides = []
        for col in range(5):
            for row in range(2):
                cx = x + col * col_width + col_width / 2
                cy = y + row * row_height + row_height / 2
                side = min(col_width, row_height)
                centers.append([cx, cy])
                sides.append(side)
        
        c_b = np.array(centers)
        s_b = np.array(sides)
        if is_feasible(c_b, s_b):
            val = objective(c_b, s_b)
            if val > best_val:
                best_val = val
                best_c = c_b.copy()
                best_s = s_b.copy()
        
        # Template C: 1 big square (0.5x0.5) in corner, 9 smaller in remaining L-shape
        # Big square: center (0.25, 0.25), side 0.5
        # Remaining regions: top strip (0, 0.5, 1, 0.5) and right strip (0.5, 0, 0.5, 0.5)
        # Top strip: 5 squares of side 0.25
        # Right strip: 4 squares of side 0.25
        centers = [[0.25, 0.25]]
        sides = [0.5]
        
        # Top strip: y from 0.5 to 1.0, x from 0 to 1.0
        # 5 squares of width 0.2, height 0.5 -> side = min(0.2, 0.5) = 0.2
        for i in range(5):
            cx = i * 0.2 + 0.1
            cy = 0.5 + 0.25
            side = min(0.2, 0.5)
            centers.append([cx, cy])
            sides.append(side)
        
        # Right strip: x from 0.5 to 1.0, y from 0 to 0.5
        # 4 squares of width 0.125, height 0.5 -> side = min(0.125, 0.5) = 0.125
        for i in range(4):
            cx = 0.5 + i * 0.125 + 0.0625
            cy = 0.25
            side = min(0.125, 0.5)
            centers.append([cx, cy])
            sides.append(side)
        
        c_c = np.array(centers)
        s_c = np.array(sides)
        if is_feasible(c_c, s_c):
            val = objective(c_c, s_c)
            if val > best_val:
                best_val = val
                best_c = c_c.copy()
                best_s = s_c.copy()
        
        # Template D: 3 large + 7 small
        # 3 squares of side ~0.33 in a row at bottom, 7 smaller above
        # Bottom row: 3 squares of side 1/3
        centers = []
        sides = []
        for i in range(3):
            cx = i / 3 + 1/6
            cy = 1/6
            side = 1/3
            centers.append([cx, cy])
            sides.append(side)
        
        # Top region: y from 1/3 to 1.0, height 2/3
        # 7 squares: try 4 in one row + 3 in another
        top_h = 2/3
        # Row 1 (bottom of top region): 4 squares
        row1_h = top_h / 2
        for i in range(4):
            cx = i * 0.25 + 0.125
            cy = 1/3 + row1_h / 2
            side = min(0.25, row1_h)
            centers.append([cx, cy])
            sides.append(side)
        
        # Row 2 (top of top region): 3 squares
        row2_h = top_h / 2
        for i in range(3):
            cx = i / 3 + 1/6
            cy = 1/3 + row1_h + row2_h / 2
            side = min(1/3, row2_h)
            centers.append([cx, cy])
            sides.append(side)
        
        c_d = np.array(centers)
        s_d = np.array(sides)
        if is_feasible(c_d, s_d):
            val = objective(c_d, s_d)
            if val > best_val:
                best_val = val
                best_c = c_d.copy()
                best_s = s_d.copy()
        
        # Template E: 4 medium + 6 small
        # 2x2 grid of 0.5x0.5 squares in bottom-left 0.5x0.5? No, that's only 1.
        # Let's do: 4 squares of side 0.25 in a 2x2 grid at bottom, 6 smaller above
        # Actually, let's try: 4 squares of side 0.3 in a 2x2 arrangement
        # 2x2 grid: each 0.3x0.3, total 0.6x0.6
        centers = []
        sides = []
        for i in range(2):
            for j in range(2):
                cx = 0.15 + i * 0.3
                cy = 0.15 + j * 0.3
                side = 0.3
                centers.append([cx, cy])
                sides.append(side)
        
        # Remaining: top strip (0, 0.6, 1, 0.4) and right strip (0.6, 0, 0.4, 0.6)
        # Top strip: 4 squares of width 0.25, height 0.4 -> side 0.25
        for i in range(4):
            cx = i * 0.25 + 0.125
            cy = 0.6 + 0.2
            side = min(0.25, 0.4)
            centers.append([cx, cy])
            sides.append(side)
        
        # Right strip: 2 squares of width 0.2, height 0.3 -> side 0.2
        for i in range(2):
            cx = 0.6 + 0.2
            cy = 0.15 + i * 0.3
            side = min(0.4, 0.3)
            centers.append([cx, cy])
            sides.append(side)
        
        c_e = np.array(centers)
        s_e = np.array(sides)
        if is_feasible(c_e, s_e):
            val = objective(c_e, s_e)
            if val > best_val:
                best_val = val
                best_c = c_e.copy()
                best_s = s_e.copy()
        
        # Template F: 10 equal squares in a spiral or other pattern
        # Let's try: 10 squares of side 0.2 in a 5x2 grid
        col_width = 0.2
        row_height = 0.5
        centers = []
        sides = []
        for col in range(5):
            for row in range(2):
                cx = col * col_width + col_width / 2
                cy = row * row_height + row_height / 2
                side = min(col_width, row_height)
                centers.append([cx, cy])
                sides.append(side)
        
        c_f = np.array(centers)
        s_f = np.array(sides)
        if is_feasible(c_f, s_f):
            val = objective(c_f, s_f)
            if val > best_val:
                best_val = val
                best_c = c_f.copy()
                best_s = s_f.copy()
        
        # Template G: Optimized 2-column layout with different row heights
        # Left column: 5 squares, Right column: 5 squares
        # But with optimized heights
        # Try: left column has 3 rows, right column has 2 rows (or vice versa)
        # Left: 3 squares, Right: 7 squares? No, let's keep it balanced
        
        # Actually, let's try a more systematic approach:
        # Search over split positions for a 2-level guillotine cut
        
        # Level 1: vertical or horizontal split
        # If vertical: left width w1, right width 1-w1
        # Left has k squares, right has 10-k squares
        # Each side gets its own sub-rectangles
        
        best_g_val = -1
        best_g_c = None
        best_g_s = None
        
        for split_type in ['v', 'h']:
            for k in range(1, 10):
                # Left/bottom has k squares, right/top has 10-k squares
                if k > 10 - k:
                    continue  # Avoid duplicates
                
                # Search over split position
                for ratio in np.linspace(0.1, 0.9, 17):
                    if split_type == 'v':
                        left_w = ratio
                        right_w = 1 - ratio
                        # Left rect: (0, 0, left_w, 1)
                        # Right rect: (left_w, 0, right_w, 1)
                        
                        # For left: k squares in left_w x 1
                        # Best is to arrange in a grid
                        # Try different row/col arrangements
                        best_left_val = -1
                        best_left_c = None
                        best_left_s = None
                        
                        for rows in range(1, k + 1):
                            cols = (k + rows - 1) // rows
                            if rows * cols < k:
                                continue
                            # Grid of rows x cols, but only k squares
                            # Cell size: left_w/cols x 1/rows
                            cell_w = left_w / cols
                            cell_h = 1 / rows
                            side = min(cell_w, cell_h)
                            
                            c_temp = []
                            s_temp = []
                            for r in range(rows):
                                for col in range(cols):
                                    if len(c_temp) >= k:
                                        break
                                    cx = col * cell_w + cell_w / 2
                                    cy = r * cell_h + cell_h / 2
                                    c_temp.append([cx, cy])
                                    s_temp.append(side)
                                if len(c_temp) >= k:
                                    break
                            
                            if len(c_temp) == k:
                                c_arr = np.array(c_temp)
                                s_arr = np.array(s_temp)
                                if is_feasible(c_arr, s_arr):
                                    val = objective(c_arr, s_arr)
                                    if val > best_left_val:
                                        best_left_val = val
                                        best_left_c = c_arr
                                        best_left_s = s_arr
                        
                        if best_left_c is None:
                            continue
                        
                        # For right: 10-k squares in right_w x 1
                        n_right = 10 - k
                        best_right_val = -1
                        best_right_c = None
                        best_right_s = None
                        
                        for rows in range(1, n_right + 1):
                            cols = (n_right + rows - 1) // rows
                            if rows * cols < n_right:
                                continue
                            cell_w = right_w / cols
                            cell_h = 1 / rows
                            side = min(cell_w, cell_h)
                            
                            c_temp = []
                            s_temp = []
                            for r in range(rows):
                                for col in range(cols):
                                    if len(c_temp) >= n_right:
                                        break
                                    cx = left_w + col * cell_w + cell_w / 2
                                    cy = r * cell_h + cell_h / 2
                                    c_temp.append([cx, cy])
                                    s_temp.append(side)
                                if len(c_temp) >= n_right:
                                    break
                            
                            if len(c_temp) == n_right:
                                c_arr = np.array(c_temp)
                                s_arr = np.array(s_temp)
                                if is_feasible(c_arr, s_arr):
                                    val = objective(c_arr, s_arr)
                                    if val > best_right_val:
                                        best_right_val = val
                                        best_right_c = c_arr
                                        best_right_s = s_arr
                        
                        if best_right_c is None:
                            continue
                        
                        total_val = best_left_val + best_right_val
                        if total_val > best_g_val:
                            best_g_val = total_val
                            best_g_c = np.vstack([best_left_c, best_right_c])
                            best_g_s = np.concatenate([best_left_s, best_right_s])
                    
                    else:  # horizontal split
                        bottom_h = ratio
                        top_h = 1 - ratio
                        # Bottom rect: (0, 0, 1, bottom_h)
                        # Top rect: (0, bottom_h, 1, top_h)
                        
                        best_bottom_val = -1
                        best_bottom_c = None
                        best_bottom_s = None
                        
                        for rows in range(1, k + 1):
                            cols = (k + rows - 1) // rows
                            if rows * cols < k:
                                continue
                            cell_w = 1 / cols
                            cell_h = bottom_h / rows
                            side = min(cell_w, cell_h)
                            
                            c_temp = []
                            s_temp = []
                            for r in range(rows):
                                for col in range(cols):
                                    if len(c_temp) >= k:
                                        break
                                    cx = col * cell_w + cell_w / 2
                                    cy = r * cell_h + cell_h / 2
                                    c_temp.append([cx, cy])
                                    s_temp.append(side)
                                if len(c_temp) >= k:
                                    break
                            
                            if len(c_temp) == k:
                                c_arr = np.array(c_temp)
                                s_arr = np.array(s_temp)
                                if is_feasible(c_arr, s_arr):
                                    val = objective(c_arr, s_arr)
                                    if val > best_bottom_val:
                                        best_bottom_val = val
                                        best_bottom_c = c_arr
                                        best_bottom_s = s_arr
                        
                        if best_bottom_c is None:
                            continue
                        
                        n_top = 10 - k
                        best_top_val = -1
                        best_top_c = None
                        best_top_s = None
                        
                        for rows in range(1, n_top + 1):
                            cols = (n_top + rows - 1) // rows
                            if rows * cols < n_top:
                                continue
                            cell_w = 1 / cols
                            cell_h = top_h / rows
                            side = min(cell_w, cell_h)
                            
                            c_temp = []
                            s_temp = []
                            for r in range(rows):
                                for col in range(cols):
                                    if len(c_temp) >= n_top:
                                        break
                                    cx = col * cell_w + cell_w / 2
                                    cy = bottom_h + r * cell_h + cell_h / 2
                                    c_temp.append([cx, cy])
                                    s_temp.append(side)
                                if len(c_temp) >= n_top:
                                    break
                            
                            if len(c_temp) == n_top:
                                c_arr = np.array(c_temp)
                                s_arr = np.array(s_temp)
                                if is_feasible(c_arr, s_arr):
                                    val = objective(c_arr, s_arr)
                                    if val > best_top_val:
                                        best_top_val = val
                                        best_top_c = c_arr
                                        best_top_s = s_arr
                        
                        if best_top_c is None:
                            continue
                        
                        total_val = best_bottom_val + best_top_val
                        if total_val > best_g_val:
                            best_g_val = total_val
                            best_g_c = np.vstack([best_bottom_c, best_top_c])
                            best_g_s = np.concatenate([best_bottom_s, best_top_s])
        
        if best_g_c is not None and best_g_val > best_val:
            best_val = best_g_val
            best_c = best_g_c.copy()
            best_s = best_g_s.copy()
        
        if best_c is None:
            return None, None
        
        return best_c, best_s

    def coordinate_ascent(c, s, max_rounds=500):
        """
        Coordinate ascent: for each square, try to increase its side length
        by repositioning within feasible space.
        """
        c = c.copy()
        s = s.copy()
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            # Randomize order to escape local optima
            order = np.random.permutation(n)
            for i in order:
                # Try to increase side of square i
                # Binary search on side length
                lo_s = s[i]
                hi_s = 1.0 + tol  # max possible side

                # Quick check: can we even grow?
                max_s_boundary = min(2 * c[i][0], 2 * (1 - c[i][0]),
                                     2 * c[i][1], 2 * (1 - c[i][1])) + 2 * tol
                hi_s = min(hi_s, max_s_boundary)
                if hi_s <= lo_s + 1e-12:
                    continue

                best_s = lo_s
                best_c = c[i].copy()

                # Binary search
                for _ in range(40):
                    mid_s = (lo_s + hi_s) / 2
                    if mid_s <= best_s + 1e-13:
                        break

                    # Check if there exists a position for square i with side mid_s
                    found = False
                    new_cx, new_cy = c[i][0], c[i][1]

                    # Try current position
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        t = (mid_s + s[j]) / 2 - tol
                        if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                            ok = False
                            break
                    if ok and new_cx - mid_s/2 >= -tol and new_cx + mid_s/2 <= 1+tol \
                       and new_cy - mid_s/2 >= -tol and new_cy + mid_s/2 <= 1+tol:
                        found = True
                    else:
                        # Try random positions
                        for _ in range(200):
                            new_cx = np.random.uniform(mid_s/2, 1 - mid_s/2)
                            new_cy = np.random.uniform(mid_s/2, 1 - mid_s/2)
                            ok = True
                            for j in range(n):
                                if j == i:
                                    continue
                                t = (mid_s + s[j]) / 2 - tol
                                if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                                    ok = False
                                    break
                            if ok:
                                found = True
                                break
                        # Also try positions near boundaries
                        if not found:
                            for dx in [0, 1]:
                                for dy in [0, 1]:
                                    new_cx = mid_s/2 + dx * (1 - mid_s)
                                    new_cy = mid_s/2 + dy * (1 - mid_s)
                                    ok = True
                                    for j in range(n):
                                        if j == i:
                                            continue
                                        t = (mid_s + s[j]) / 2 - tol
                                        if abs(new_cx - c[j][0]) < t and abs(new_cy - c[j][1]) < t:
                                            ok = False
                                            break
                                    if ok:
                                        found = True
                                        break
                                if found:
                                    break

                    if found:
                        best_s = mid_s
                        best_c = np.array([new_cx, new_cy])
                        lo_s = mid_s
                    else:
                        hi_s = mid_s

                if best_s > s[i] + 1e-10:
                    s[i] = best_s
                    c[i] = best_c
                    improved = True

            if not improved:
                break

        return c, s

    def local_search(c, s, max_iters=2000):
        """
        More aggressive local search: try perturbing positions and sizes.
        """
        c = c.copy()
        s = s.copy()
        best_val = objective(c, s)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Pick a random square to perturb
            i = np.random.randint(n)
            old_c = c[i].copy()
            old_s = s[i]

            # Random perturbation
            ds = np.random.uniform(-0.05, 0.1)
            new_s = max(0.01, min(1.0, old_s + ds))
            dcx = np.random.uniform(-0.1, 0.1)
            dcy = np.random.uniform(-0.1, 0.1)
            new_cx = np.clip(old_c[0] + dcx, new_s/2, 1 - new_s/2)
            new_cy = np.clip(old_c[1] + dcy, new_s/2, 1 - new_s/2)

            c[i] = [new_cx, new_cy]
            s[i] = new_s

            if is_feasible(c, s):
                new_val = objective(c, s)
                if new_val > best_val:
                    best_val = new_val
                else:
                    # Revert
                    c[i] = old_c
                    s[i] = old_s
            else:
                # Revert
                c[i] = old_c
                s[i] = old_s

        return c, s

    # Multi-start optimization
    best_c = None
    best_s = None
    best_val = -1

    # Start 1: Guillotine packing
    g_c, g_s = guillotine_optimize()
    if g_c is not None and is_feasible(g_c, g_s):
        val = objective(g_c, g_s)
        if val > best_val:
            best_val = val
            best_c = g_c.copy()
            best_s = g_s.copy()
    
    # Refine guillotine result with coordinate ascent
    if best_c is not None:
        c_opt, s_opt = coordinate_ascent(best_c, best_s, max_rounds=300)
        if is_feasible(c_opt, s_opt):
            val = objective(c_opt, s_opt)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
                best_s = s_opt.copy()

    # Start 2: Analytical construction
    centers_init = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
    ], dtype=np.float64)
    sides_init = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    c_opt, s_opt = coordinate_ascent(centers_init, sides_init, max_rounds=300)
    if is_feasible(c_opt, s_opt):
        val = objective(c_opt, s_opt)
        if val > best_val:
            best_val = val
            best_c = c_opt.copy()
            best_s = s_opt.copy()

    # More random starts
    np.random.seed(42)
    for start in range(30):
        if time.time() > deadline:
            break
        # Random feasible config
        s_rand = np.random.uniform(0.01, 0.3, size=n)
        c_rand = np.zeros((n, 2))
        feasible = True
        for i in range(n):
            placed = False
            for _ in range(200):
                cx = np.random.uniform(s_rand[i]/2, 1 - s_rand[i]/2)
                cy = np.random.uniform(s_rand[i]/2, 1 - s_rand[i]/2)
                c_rand[i] = [cx, cy]
                ok = True
                for j in range(i):
                    t = (s_rand[i] + s_rand[j]) / 2 - tol
                    if abs(cx - c_rand[j][0]) < t and abs(cy - c_rand[j][1]) < t:
                        ok = False
                        break
                if ok:
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        
        if not feasible or not is_feasible(c_rand, s_rand):
            continue
            
        c_opt, s_opt = coordinate_ascent(c_rand, s_rand, max_rounds=200)
        if is_feasible(c_opt, s_opt):
            val = objective(c_opt, s_opt)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
                best_s = s_opt.copy()

        # Additional local search
        c_ls, s_ls = local_search(c_opt, s_opt, max_iters=500)
        if is_feasible(c_ls, s_ls):
            val = objective(c_ls, s_ls)
            if val > best_val:
                best_val = val
                best_c = c_ls.copy()
                best_s = s_ls.copy()

    if best_c is None:
        # Fallback
        best_c = centers_init.copy()
        best_s = sides_init.copy()

    # Final verification
    if not is_feasible(best_c, best_s):
        best_c = centers_init.copy()
        best_s = sides_init.copy()

    best_s = np.maximum(best_s, 1e-9)

    return best_c, best_s


# EVOLVE-BLOCK-END
