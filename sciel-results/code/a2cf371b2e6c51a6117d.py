import numpy as np
import time
import itertools
from scipy.optimize import linear_sum_assignment

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_strip_layout(n_sq, width, height):
    """
    Solve for max sum of sides for n squares packed in a strip of given width/height.
    This assumes a "guillotine" structure where squares are arranged in rows/columns.
    For a simple strip of n squares side-by-side:
    If width >= height:
        Each square i has side s_i.
        Constraint: sum(s_i) <= width (if they are in a row) OR s_i <= height (if limited by height).
        Actually, a general strip packing is: partition into rows.
        Let's simplify: try all partitions of n into k rows.
        For each row r with n_r squares, the height of the row is max(s_i for i in row).
        The width of the row is sum(s_i for i in row).
        Constraints:
        1. sum(height_of_row_r) <= height
        2. For each row r: sum(s_i) <= width AND max(s_i) <= height_of_row_r
        3. s_i > 0
        
        To maximize sum(s_i), we can assume all squares in a row have the same side length?
        No, that's suboptimal. But if we fix the row structure, the problem is:
        Maximize sum(s_i)
        Subject to:
        For each row r:
           sum_{i in r} s_i <= width
           max_{i in r} s_i <= H_r
        sum_r H_r <= height
        s_i >= 0
        
        This is a linear programming problem if H_r are variables.
        Let's use a simpler heuristic: 
        For a given partition of n into rows, assume all squares in a row have equal side s_r.
        Then row width = n_r * s_r <= width => s_r <= width / n_r
        Row height = s_r
        Total height = sum(s_r) <= height
        Maximize sum(n_r * s_r)
        
        This reduces to:
        Maximize sum(n_r * s_r)
        s.t. s_r <= width / n_r
             sum(s_r) <= height
             s_r >= 0
             
        This is a simple LP. The optimal solution is to fill s_r as much as possible.
        Since objective is linear and constraints are box + sum, the optimum is at a vertex.
        We can just iterate: 
        1. Start with s_r = width / n_r for all r.
        2. If sum(s_r) > height, scale down? No, that's not optimal if some rows are "cheaper" in height.
        Actually, the constraint is sum(s_r) <= height.
        To maximize sum(n_r s_r), we should prioritize rows with larger n_r?
        Let's just use scipy linprog or a simple greedy since n is small.
        Actually, we can just try:
        s_r = min(width/n_r, remaining_height_budget_allocated)
        
        Let's implement a more general solver for "Rows of equal-size squares".
        """
    if n_sq == 0:
        return 0.0, []
        
    # Generate all integer partitions of n_sq
    # Partition n into k parts, each part is the number of squares in a row.
    # Order matters for the geometry? No, rows are independent.
    # But we need to consider all compositions (ordered partitions) because 
    # the "width" constraint applies to each row, and "height" is sum.
    # However, since rows are stacked, the order doesn't change the sum of heights.
    # So we just need the multiset of row sizes.
    
    def get_compositions(n, max_part=None):
        if max_part is None:
            max_part = n
        if n == 0:
            yield []
            return
        for i in range(1, max_part + 1):
            if i <= n:
                for rest in get_compositions(n - i, max_part):
                    yield [i] + rest
                    
    best_val = 0.0
    best_sides = []
    
    # Limit search to reasonable number of rows (1 to n_sq)
    # Number of compositions of 10 is 2^9 = 512. This is small.
    for comp in get_compositions(n_sq):
        # comp is a list of row sizes, e.g., [3, 2, 5]
        k = len(comp)
        # Variables: s_1, ..., s_k (side length of squares in each row)
        # Objective: max sum(comp[r] * s_r)
        # Constraints:
        # 1. s_r <= width / comp[r]
        # 2. sum(s_r) <= height
        # 3. s_r >= 0
        
        # This is a standard LP. We can solve it analytically for small k.
        # Since the objective coefficients are comp[r], and the "cost" in the sum constraint is 1,
        # we should allocate height to rows with highest comp[r] first?
        # Let's verify:
        # Max sum(c_r s_r) s.t. sum(s_r) <= H, s_r <= U_r.
        # This is like filling a knapsack where item r has value c_r and weight 1, capacity H, limit U_r.
        # Greedy: Sort rows by c_r (descending). Fill s_r up to U_r until H is exhausted.
        
        rows = sorted(range(k), key=lambda r: comp[r], reverse=True)
        remaining_h = height
        s_vals = [0.0] * k
        for r in rows:
            max_s = width / comp[r]
            take = min(max_s, remaining_h)
            s_vals[r] = take
            remaining_h -= take
            if remaining_h <= 1e-9:
                break
                
        val = sum(comp[r] * s_vals[r] for r in range(k))
        if val > best_val:
            best_val = val
            # Map back to individual squares
            # We need to return the side for each square in the original order?
            # The caller will handle placement. We just need the side lengths.
            # But wait, the squares in a row are identical in this model.
            # We need to return a list of side lengths of length n_sq.
            curr_idx = 0
            sides_list = []
            for r in range(k):
                for _ in range(comp[r]):
                    sides_list.append(s_vals[r])
            if len(sides_list) == n_sq:
                best_sides = sides_list
                
    return best_val, best_sides

def _solve_recursive_layout(n_sq, rect_width, rect_height, depth=0):
    """
    Recursively split the rectangle into two parts (guillotine cut).
    Try horizontal and vertical splits.
    For each split, distribute squares between the two sub-rectangles.
    """
    if n_sq == 0:
        return 0.0, []
    if depth > 4: # Prevent infinite recursion, 10 squares need at most ~5 levels
        # Fallback to strip layout
        return _solve_strip_layout(n_sq, rect_width, rect_height)
        
    best_val = 0.0
    best_sides = []
    
    # Option 1: Treat as a single strip (no split)
    val_strip, sides_strip = _solve_strip_layout(n_sq, rect_width, rect_height)
    if val_strip > best_val:
        best_val = val_strip
        best_sides = sides_strip
        
    # Option 2: Split
    # We need to partition the set of n_sq squares into two subsets S1 and S2.
    # Since squares are indistinguishable in terms of "identity" for the optimization 
    # (we just want the multiset of side lengths), we can just choose k1 for S1 and k2=n-k1 for S2.
    # However, the specific assignment of which square goes where matters for the final coordinates,
    # but for the *value* (sum of sides), we just need the optimal side lengths for each subset.
    
    for k1 in range(1, n_sq):
        k2 = n_sq - k1
        
        # Vertical split: width is split into w1, w2. height is same.
        # We can choose the split position.
        # Let w1 be the width of left part. w2 = rect_width - w1.
        # The optimal w1 depends on k1, k2 and the resulting side lengths.
        # This is a 1D optimization for w1.
        # Let f(w1) = max_val(k1, w1, rect_height) + max_val(k2, rect_width-w1, rect_height)
        # We can sample w1 or use a ternary search? 
        # The function is likely concave. Let's use a grid search + refinement.
        
        # Similarly for horizontal split.
        
        # To keep it fast, we will sample a few split ratios.
        # Better: The optimal split often aligns with the aspect ratios of the sub-problems.
        # For now, use a coarse grid for the split parameter.
        
        # Vertical
        best_v_val = 0.0
        best_v_sides = []
        best_v_w1 = 0
        
        # Sample split points
        # The width allocated to a group should be proportional to the "demand".
        # Let's just iterate over possible integer splits of width? No, continuous.
        # Let's try 20 split points.
        for i in range(1, 20):
            w1 = rect_width * i / 20.0
            w2 = rect_width - w1
            if w1 < 1e-9 or w2 < 1e-9:
                continue
            v1, s1 = _solve_recursive_layout(k1, w1, rect_height, depth + 1)
            v2, s2 = _solve_recursive_layout(k2, w2, rect_height, depth + 1)
            total_v = v1 + v2
            if total_v > best_v_val:
                best_v_val = total_v
                best_v_sides = s1 + s2
                best_v_w1 = w1
        
        if best_v_val > best_val:
            best_val = best_v_val
            best_sides = best_v_sides
            
        # Horizontal
        best_h_val = 0.0
        best_h_sides = []
        for i in range(1, 20):
            h1 = rect_height * i / 20.0
            h2 = rect_height - h1
            if h1 < 1e-9 or h2 < 1e-9:
                continue
            v1, s1 = _solve_recursive_layout(k1, rect_width, h1, depth + 1)
            v2, s2 = _solve_recursive_layout(k2, rect_width, h2, depth + 1)
            total_h = v1 + v2
            if total_h > best_h_val:
                best_h_val = total_h
                best_h_sides = s1 + s2
                
        if best_h_val > best_val:
            best_val = best_h_val
            best_sides = best_h_sides

    return best_val, best_sides

def _place_squares_guillotine(n_sq, rect_width, rect_height, start_x, start_y):
    """
    Recursively places squares and returns (centers, sides).
    This mirrors the logic in _solve_recursive_layout but actually positions them.
    """
    if n_sq == 0:
        return [], []
    if rect_width < 1e-9 or rect_height < 1e-9:
        return [], []
        
    # Try strip layout first (fastest)
    # We need to know *which* layout was best.
    # To avoid recomputing, we should cache or just re-evaluate.
    # Given n=10, re-evaluation is cheap.
    
    best_val = -1
    best_centers = []
    best_sides = []
    
    # 1. Try Strip Layout
    val_strip, sides_strip = _solve_strip_layout(n_sq, rect_width, rect_height)
    if val_strip > best_val:
        best_val = val_strip
        # Place squares in a row
        # sides_strip has length n_sq
        curr_x = start_x
        centers = []
        for i in range(n_sq):
            s = sides_strip[i]
            if s > 1e-9:
                cx = curr_x + s / 2
                cy = start_y + s / 2
                centers.append([cx, cy])
            else:
                centers.append([start_x, start_y]) # Dummy
            curr_x += s
        best_centers = centers
        best_sides = sides_strip
        
    # 2. Try Recursive Splits
    for k1 in range(1, n_sq):
        k2 = n_sq - k1
        
        # Vertical Split
        # We need to find the best w1.
        # We can reuse the search from _solve_recursive_layout
        # For simplicity and correctness, let's just try the same grid of splits.
        for i in range(1, 20):
            w1 = rect_width * i / 20.0
            w2 = rect_width - w1
            if w1 < 1e-9 or w2 < 1e-9:
                continue
            
            # Solve sub-problems
            # Note: _solve_recursive_layout returns optimal value and sides, but not coordinates.
            # We need to call a version that returns coordinates.
            # This is getting complex. Let's simplify:
            # We will just use the Strip Layout for the sub-rectangles in the recursive call.
            # This restricts us to "strip-of-strips" (guillotine) layouts.
            
            # Left part
            v1, s1 = _solve_strip_layout(k1, w1, rect_height)
            # Right part
            v2, s2 = _solve_strip_layout(k2, w2, rect_height)
            
            total_v = v1 + v2
            if total_v > best_val:
                best_val = total_v
                # Place left
                c1, s1_placed = _place_squares_guillotine(k1, w1, rect_height, start_x, start_y)
                # Place right
                c2, s2_placed = _place_squares_guillotine(k2, w2, rect_height, start_x + w1, start_y)
                best_centers = c1 + c2
                best_sides = s1_placed + s2_placed
                
        # Horizontal Split
        for i in range(1, 20):
            h1 = rect_height * i / 20.0
            h2 = rect_height - h1
            if h1 < 1e-9 or h2 < 1e-9:
                continue
            
            # Top part (let's say)
            v1, s1 = _solve_strip_layout(k1, rect_width, h1)
            # Bottom part
            v2, s2 = _solve_strip_layout(k2, rect_width, h2)
            
            total_h = v1 + v2
            if total_h > best_val:
                best_val = total_h
                # Place top
                c1, s1_placed = _place_squares_guillotine(k1, rect_width, h1, start_x, start_y)
                # Place bottom
                c2, s2_placed = _place_squares_guillotine(k2, rect_width, h2, start_x, start_y + h1)
                best_centers = c1 + c2
                best_sides = s1_placed + s2_placed
                
    return best_centers, best_sides

def _refine_continuous(centers, sides, tol=1e-7, time_limit=10):
    """
    Take a valid guillotine layout and try to improve it by continuous relaxation.
    This allows "sliding" squares to fill gaps that guillotine cuts might leave.
    """
    n = len(sides)
    if n == 0:
        return centers, sides
    
    start = time.time()
    centers = np.array(centers, dtype=np.float64)
    sides = np.array(sides, dtype=np.float64)
    
    # Simple local search: try to expand each square slightly if it doesn't collide.
    # Or move them towards the center of the available space.
    # Let's do a few iterations of "expand if possible".
    
    for _ in range(50):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            # Try to expand s_i by a small amount
            eps = 1e-4
            new_s = sides[i] + eps
            if new_s > 1.0:
                continue
            
            # Check if it fits in bounds
            cx, cy = centers[i]
            if cx - new_s/2 < -tol or cx + new_s/2 > 1 + tol:
                # Try moving center to fit
                new_cx = np.clip(cx, new_s/2, 1 - new_s/2)
                if abs(new_cx - cx) > 1e-9:
                    # Check if move causes collision
                    # Simplified: just check bounds for now, collision check is expensive
                    # For refinement, we assume guillotine layout is safe, 
                    # so expansion is safe if it doesn't cross the "cut" line.
                    # This is hard to check without storing topology.
                    # So we only expand if it stays within the original bounding box of the square?
                    # No, that defeats the purpose.
                    
                    # Let's just skip expansion if it hits bounds, to be safe.
                    continue
            if cy - new_s/2 < -tol or cy + new_s/2 > 1 + tol:
                continue
                
            # Check collisions with neighbors
            valid = True
            for j in range(n):
                if i == j:
                    continue
                t = (new_s + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    valid = False
                    break
            
            if valid:
                sides[i] = new_s
                improved = True
                
        if not improved:
            break
            
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    Strategy: Guillotine decomposition search + local refinement.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 570
    
    # 1. Find best guillotine layout
    best_val = 0.0
    best_centers = []
    best_sides = []
    
    # The recursive placement function handles the search.
    # It might be slow if it explores too many splits.
    # Let's limit the depth or use the direct solver for n=10.
    # _place_squares_guillotine is O(2^n * grid_size). 
    # For n=10, 2^9=512 splits. Grid 20. 512*20*10 (recursion) ~ 100k ops. Fast.
    
    try:
        centers, sides = _place_squares_guillotine(n, 1.0, 1.0, 0.0, 0.0)
        if len(centers) == n:
            val = np.sum(sides)
            if val > best_val:
                best_val = val
                best_centers = centers
                best_sides = sides
    except Exception:
        pass
    
    # 2. Refine the best guillotine layout
    if len(best_centers) == n:
        try:
            c_ref, s_ref = _refine_continuous(best_centers, best_sides, tol, time_limit=30)
            if _is_valid(c_ref, s_ref, tol):
                val_ref = np.sum(s_ref)
                if val_ref > best_val:
                    best_val = val_ref
                    best_centers = c_ref
                    best_sides = s_ref
        except Exception:
            pass
    
    # 3. Fallback: Simple grid if something went wrong
    if len(best_centers) != n or not _is_valid(np.array(best_centers), np.array(best_sides), tol):
        # 5x2 grid of 0.18 squares?
        # 5 columns: 5*0.18 = 0.9. 2 rows: 2*0.18 = 0.36.
        # We can make them bigger.
        # 5 cols: s=0.19 -> 0.95. 2 rows: 0.38.
        # 10 squares.
        c = np.array([[0.095 + 0.19*i, 0.095 + 0.19*j] for i in range(5) for j in range(2)])
        s = np.full(10, 0.19)
        if _is_valid(c, s, tol):
            best_centers = c
            best_sides = s
            best_val = np.sum(s)
        else:
            # Tiny squares
            best_centers = np.zeros((10, 2))
            best_sides = np.full(10, 0.01)
            
    return np.array(best_centers), np.array(best_sides)

# EVOLVE-BLOCK-END
