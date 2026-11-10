import numpy as np
import time
import itertools

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

def _get_compositions(n):
    """Generate all compositions of n (ordered integer partitions)."""
    if n == 0:
        yield []
        return
    for i in range(1, n + 1):
        for rest in _get_compositions(n - i):
            yield [i] + rest

def _solve_shelf_layout(comp):
    """
    Solve for optimal side lengths for a shelf layout defined by 'comp' (row sizes).
    Maximize sum(n_r * s_r)
    Subject to:
      s_r <= 1.0 / n_r  (width constraint for row r)
      sum(s_r) <= 1.0   (height constraint for total stack)
      s_r >= 0
    """
    k = len(comp)
    # Greedy allocation: prioritize rows with larger n_r (higher value per unit height)
    # Sort indices by n_r descending
    indices = sorted(range(k), key=lambda r: comp[r], reverse=True)
    
    s_vals = [0.0] * k
    remaining_h = 1.0
    
    for r in indices:
        max_s = 1.0 / comp[r]
        take = min(max_s, remaining_h)
        if take < 1e-9:
            break
        s_vals[r] = take
        remaining_h -= take
        
    # Calculate value and map back to individual squares
    val = sum(comp[r] * s_vals[r] for r in range(k))
    
    sides_list = []
    for r in range(k):
        for _ in range(comp[r]):
            sides_list.append(s_vals[r])
            
    return val, sides_list

def _generate_shelf_layout(comp, sides_list):
    """
    Generate centers and sides for a specific shelf layout.
    comp: list of row sizes
    sides_list: list of side lengths corresponding to squares in order
    """
    centers = []
    curr_y = 0.0
    idx = 0
    
    for r, n_r in enumerate(comp):
        # Get sides for this row
        row_sides = sides_list[idx:idx + n_r]
        s_row = row_sides[0] # All squares in a row have same side in this model
        
        # Place squares in this row
        curr_x = 0.0
        for i in range(n_r):
            cx = curr_x + s_row / 2
            cy = curr_y + s_row / 2
            centers.append([cx, cy])
            curr_x += s_row
            
        curr_y += s_row
        idx += n_r
        
    return np.array(centers), np.array(sides_list)

def _local_refine(centers, sides, tol=1e-7, time_limit=5.0):
    """
    Refine a valid layout by trying to expand each square as much as possible.
    Uses a simple iterative approach: try to increase s_i by eps, check validity.
    If valid, keep. If not, try smaller increments or move center.
    For speed and safety, we only try expansion in small steps.
    """
    n = len(sides)
    if n == 0:
        return centers, sides
    
    start = time.time()
    centers = np.array(centers, dtype=np.float64)
    sides = np.array(sides, dtype=np.float64)
    
    # We will iterate a few times, trying to grow each square.
    # To avoid O(N^2 * iterations) explosion, we limit iterations.
    for _ in range(20):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            # Try to expand s_i
            # Max possible size is limited by bounds and neighbors
            # Simple strategy: try increasing by 1e-4, 1e-5, etc?
            # Or just try a few random small expansions?
            # Let's try a binary search for max expansion for square i,
            # but that's expensive.
            # Simpler: try expanding by 1e-4. If valid, do it.
            eps = 1e-4
            new_s = sides[i] + eps
            if new_s > 1.0:
                continue
            
            # Check bounds
            cx, cy = centers[i]
            # If it hits boundary, try moving center to center of available space?
            # For simplicity, if it hits boundary, skip expansion for this iteration
            # or just clip? Clipping might cause overlap.
            # Let's just check if the expanded square is within bounds.
            if cx - new_s/2 < -tol or cx + new_s/2 > 1 + tol:
                continue
            if cy - new_s/2 < -tol or cy + new_s/2 > 1 + tol:
                continue
            
            # Check collisions
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
    Strategy: Enumerate all shelf layouts, find best, then refine.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 570
    
    best_val = 0.0
    best_centers = None
    best_sides = None
    
    # 1. Enumerate all shelf layouts (compositions of 10)
    # There are 2^9 = 512 compositions.
    # For each, solve the LP (greedy) and generate layout.
    
    for comp in _get_compositions(n):
        if time.time() > deadline - 60:
            break
            
        val, sides_list = _solve_shelf_layout(comp)
        if val <= best_val:
            continue
            
        centers, sides = _generate_shelf_layout(comp, sides_list)
        
        # Verify validity (should be valid by construction, but check tol)
        if _is_valid(centers, sides, tol):
            best_val = val
            best_centers = centers
            best_sides = sides
        else:
            # This shouldn't happen for shelf layouts, but just in case
            pass
            
    # 2. Refine the best layout found
    if best_centers is not None:
        try:
            c_ref, s_ref = _local_refine(best_centers, best_sides, tol, time_limit=30)
            if _is_valid(c_ref, s_ref, tol):
                val_ref = np.sum(s_ref)
                if val_ref > best_val:
                    best_val = val_ref
                    best_centers = c_ref
                    best_sides = s_ref
        except Exception:
            pass
            
    # 3. Fallback if something went wrong
    if best_centers is None:
        # Simple grid: 5x2 of 0.19 squares
        c = np.array([[0.095 + 0.19*i, 0.095 + 0.19*j] for i in range(5) for j in range(2)])
        s = np.full(10, 0.19)
        if _is_valid(c, s, tol):
            best_centers = c
            best_sides = s
        else:
            best_centers = np.zeros((10, 2))
            best_sides = np.full(10, 0.01)
            
    return best_centers, best_sides
