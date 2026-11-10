# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    # No overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _max_side_for_i(i, centers, sides, tol=1e-7):
    """
    Given all other squares fixed, compute the maximum side length for square i
    such that it fits inside [0,1]^2 and does not overlap any other square.
    The center is also optimized (within the feasible region for that side).
    """
    n = len(sides)
    # Binary search on side length s_i
    # For a given s_i, the center must be in [s_i/2, 1-s_i/2]^2
    # and for each j, NOT (|cx_i - cx_j| < (s_i+s_j)/2 - tol AND |cy_i - cy_j| < (s_i+s_j)/2 - tol)
    
    def feasible_side(s_i):
        if s_i <= 0 or s_i >= 1:
            return False
        lo_x = s_i / 2
        hi_x = 1 - s_i / 2
        lo_y = s_i / 2
        hi_y = 1 - s_i / 2
        if lo_x >= hi_x or lo_y >= hi_y:
            return False
        
        # For each other square j, the forbidden region for center_i is a rectangle:
        # |cx_i - cx_j| < (s_i+s_j)/2 - tol  AND  |cy_i - cy_j| < (s_i+s_j)/2 - tol
        # i.e., cx_i in (cx_j - t_j, cx_j + t_j) AND cy_i in (cy_j - t_j, cy_j + t_j)
        # where t_j = (s_i + s_j)/2 - tol
        
        # We need to check if the rectangle [lo_x, hi_x] x [lo_y, hi_y] has any point
        # that is NOT in the union of all forbidden rectangles.
        # For small n, we can check: is the feasible region non-empty?
        # The feasible region is [lo_x,hi_x]x[lo_y,hi_y] minus union of forbidden rects.
        # A point (x,y) is feasible if for ALL j, x is outside (cx_j - t_j, cx_j + t_j) 
        # OR y is outside (cy_j - t_j, cy_j + t_j).
        
        # Strategy: check if there exists a feasible point.
        # We can try: for each j, the constraint is a "cross" shape exclusion.
        # Actually, let's just check if the feasible rectangle is fully covered by forbidden regions.
        # For n=10, we can use a simple approach: check candidate points.
        
        # Better approach: The feasible set is non-empty iff there's a point (x,y) in the bounding
        # box such that for every j, x <= cx_j - t_j or x >= cx_j + t_j or y <= cy_j - t_j or y >= cy_j + t_j.
        
        # Let's collect all critical x-coordinates and y-coordinates.
        x_cands = [lo_x, hi_x]
        y_cands = [lo_y, hi_y]
        for j in range(n):
            if j == i:
                continue
            t_j = (s_i + sides[j]) / 2 - tol
            x_cands.extend([centers[j, 0] - t_j, centers[j, 0] + t_j])
            y_cands.extend([centers[j, 1] - t_j, centers[j, 1] + t_j])
        
        x_cands = [max(lo_x, min(hi_x, x)) for x in x_cands]
        y_cands = [max(lo_y, min(hi_y, y)) for y in y_cands]
        
        # Also add midpoints of intervals for robustness
        x_sorted = sorted(set(x_cands))
        y_sorted = sorted(set(y_cands))
        
        # Check all pairs of candidate points (plus small offsets)
        for x in x_cands:
            for y in y_cands:
                if lo_x - 1e-12 <= x <= hi_x + 1e-12 and lo_y - 1e-12 <= y <= hi_y + 1e-12:
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        t_j = (s_i + sides[j]) / 2 - tol
                        if abs(x - centers[j, 0]) < t_j - 1e-12 and abs(y - centers[j, 1]) < t_j - 1e-12:
                            ok = False
                            break
                    if ok:
                        return True
        
        # Also check midpoints of the bounding box
        for x in [lo_x, hi_x, (lo_x+hi_x)/2]:
            for y in [lo_y, hi_y, (lo_y+hi_y)/2]:
                if lo_x - 1e-12 <= x <= hi_x + 1e-12 and lo_y - 1e-12 <= y <= hi_y + 1e-12:
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        t_j = (s_i + sides[j]) / 2 - tol
                        if abs(x - centers[j, 0]) < t_j - 1e-12 and abs(y - centers[j, 1]) < t_j - 1e-12:
                            ok = False
                            break
                    if ok:
                        return True
        
        return False
    
    # Binary search for max feasible side
    lo = 0.001
    hi = 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if feasible_side(mid):
            lo = mid
        else:
            hi = mid
    
    return lo


def _find_center_for_side(i, s_i, centers, sides, tol=1e-7):
    """Find a valid center for square i with side s_i, given other squares fixed."""
    n = len(sides)
    lo_x = s_i / 2
    hi_x = 1 - s_i / 2
    lo_y = s_i / 2
    hi_y = 1 - s_i / 2
    
    # Try to find a center that doesn't overlap any other square
    # Start with the center that maximizes distance to all others
    best_center = None
    best_score = -1
    
    # Generate candidate points
    x_cands = [lo_x, hi_x, (lo_x+hi_x)/2]
    y_cands = [lo_y, hi_y, (lo_y+hi_y)/2]
    
    for j in range(n):
        if j == i:
            continue
        t_j = (s_i + sides[j]) / 2 - tol
        # Try to be just outside the forbidden region
        x_cands.extend([centers[j, 0] - t_j, centers[j, 0] + t_j])
        y_cands.extend([centers[j, 1] - t_j, centers[j, 1] + t_j])
    
    x_cands = [max(lo_x, min(hi_x, x)) for x in x_cands]
    y_cands = [max(lo_y, min(hi_y, y)) for y in y_cands]
    
    for x in x_cands:
        for y in y_cands:
            if lo_x - 1e-12 <= x <= hi_x + 1e-12 and lo_y - 1e-12 <= y <= hi_y + 1e-12:
                ok = True
                min_dist = float('inf')
                for j in range(n):
                    if j == i:
                        continue
                    t_j = (s_i + sides[j]) / 2 - tol
                    if abs(x - centers[j, 0]) < t_j - 1e-12 and abs(y - centers[j, 1]) < t_j - 1e-12:
                        ok = False
                        break
                    # Score: minimum separation from forbidden regions
                    sep_x = abs(abs(x - centers[j, 0]) - t_j)
                    sep_y = abs(abs(y - centers[j, 1]) - t_j)
                    min_dist = min(min_dist, sep_x + sep_y)
                if ok:
                    if min_dist > best_score:
                        best_score = min_dist
                        best_center = (x, y)
    
    if best_center is None:
        # Fallback: try random points
        rng = np.random.default_rng(0)
        for _ in range(1000):
            x = rng.uniform(lo_x, hi_x)
            y = rng.uniform(lo_y, hi_y)
            ok = True
            for j in range(n):
                if j == i:
                    continue
                t_j = (s_i + sides[j]) / 2 - tol
                if abs(x - centers[j, 0]) < t_j - 1e-12 and abs(y - centers[j, 1]) < t_j - 1e-12:
                    ok = False
                    break
            if ok:
                best_center = (x, y)
                break
    
    if best_center is None:
        return None
    
    return best_center


def _greedy_optimize(centers, sides, tol=1e-7, max_rounds=200, time_limit=None):
    """
    Deterministic greedy optimization:
    - In each round, for each square i (in order), maximize its side given others fixed.
    - Then, do a position optimization pass to create room for further growth.
    """
    n = len(sides)
    centers = centers.copy().astype(np.float64)
    sides = sides.copy().astype(np.float64)
    
    if not _is_valid(centers, sides, tol):
        # Try to repair
        for i in range(n):
            # Shrink to minimum
            sides[i] = 0.01
            centers[i] = [0.5, 0.5]
        return centers, sides
    
    start_time = time.time()
    
    for rnd in range(max_rounds):
        if time_limit is not None and time.time() > time_limit:
            break
        
        improved = False
        
        # Phase 1: Maximize each square's side
        for i in range(n):
            old_s = sides[i]
            new_s = _max_side_for_i(i, centers, sides, tol)
            new_s = max(new_s, 0.001)
            
            if new_s > old_s + 1e-9:
                # Find a valid center for the new side
                center = _find_center_for_side(i, new_s, centers, sides, tol)
                if center is not None:
                    centers[i] = center
                    sides[i] = new_s
                    improved = True
        
        # Phase 2: Position optimization - try to move squares to create more room
        # For each pair of squares, try to push them apart
        for i in range(n):
            for j in range(i+1, n):
                # Try small perturbations to both
                for dx in [-0.01, 0, 0.01]:
                    for dy in [-0.01, 0, 0.01]:
                        new_centers = centers.copy()
                        new_sides = sides.copy()
                        new_centers[i, 0] += dx
                        new_centers[i, 1] += dy
                        new_centers[j, 0] -= dx
                        new_centers[j, 1] -= dy
                        
                        # Clamp
                        for k in [i, j]:
                            new_sides[k] = max(0.001, min(1.0, new_sides[k]))
                            new_centers[k, 0] = max(new_sides[k]/2, min(1-new_sides[k]/2, new_centers[k, 0]))
                            new_centers[k, 1] = max(new_sides[k]/2, min(1-new_sides[k]/2, new_centers[k, 1]))
                        
                        if _is_valid(new_centers, new_sides, tol):
                            # Check if this creates room for growth
                            total_potential = 0
                            for k in range(n):
                                total_potential += _max_side_for_i(k, new_centers, new_sides, tol)
                            if total_potential > _objective(centers, sides) + 1e-6:
                                centers = new_centers
                                sides = new_sides
                                improved = True
        
        if not improved:
            break
    
    return centers, sides


def _seed_grid():
    """3x3 grid + 1 extra, or 4x4 grid subset."""
    n = 10
    # 4x4 grid, use 10 cells
    side = 0.25
    centers = []
    sides = []
    # Use first 10 cells of a 4x4 grid
    for idx in range(10):
        col = idx % 4
        row = idx // 4
        cx = (col + 0.5) * side
        cy = (row + 0.5) * side
        centers.append([cx, cy])
        sides.append(side)
    return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)


def _seed_heterogeneous1():
    """
    One large square in corner, rest smaller.
    Layout: Large square (0.4) at bottom-left, then fill remaining space.
    """
    n = 10
    centers = []
    sides = []
    
    # Large square
    s0 = 0.4
    centers.append([s0/2, s0/2])
    sides.append(s0)
    
    # Remaining space: [0.4, 1] x [0, 1] and [0, 0.4] x [0.4, 1]
    # Fill [0.4, 1] x [0, 1] with squares of side 0.3 (3 cols x 3 rows = 9, but we need 9 more)
    # Actually [0.4,1] is 0.6 wide, [0,1] is 1 tall.
    # Let's do: right strip [0.4,1]x[0,1]: 2 cols of 0.3, 3 rows of 0.333
    # Top strip [0,0.4]x[0.4,1]: 1 col of 0.4, 1 row of 0.6
    
    # Right strip: 2 cols x 3 rows, side = min(0.6/2, 1/3) = 0.3
    s1 = 0.3
    for r in range(3):
        for c in range(2):
            cx = 0.4 + (c + 0.5) * s1
            cy = (r + 0.5) * s1
            centers.append([cx, cy])
            sides.append(s1)
    
    # Top-left: [0, 0.4] x [0.4, 1], side = 0.4 (but height is 0.6, so can be larger)
    # Actually [0,0.4] width=0.4, [0.4,1] height=0.6. Max square = 0.4
    s2 = 0.4
    centers.append([s2/2, 0.4 + (1-s2)/2])
    sides.append(s2)
    
    # That's 1 + 6 + 1 = 8. Need 2 more.
    # The right strip has [0.4,1]x[0,1] with 2x3=6 squares of 0.3, total width=0.6, height=0.9.
    # Remaining: [0.4,1]x[0.9,1] is 0.6 x 0.1 - too small for 0.3
    # And [0.4,1]x[0,0.9] is filled.
    # Let me redo: use a different layout.
    
    # Simpler: just return what we have if it's 10
    if len(centers) == 10:
        return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)
    
    # Fallback to grid
    return _seed_grid()


def _seed_heterogeneous2():
    """
    Two large squares in opposite corners, rest medium.
    """
    n = 10
    centers = []
    sides = []
    
    # Bottom-left: 0.35
    s0 = 0.35
    centers.append([s0/2, s0/2])
    sides.append(s0)
    
    # Top-right: 0.35
    s1 = 0.35
    centers.append([1 - s1/2, 1 - s1/2])
    sides.append(s1)
    
    # Fill remaining with 0.2 squares in a grid-like pattern
    # Available: lots of space
    # Just place 8 more squares of side 0.2 in available spots
    s2 = 0.2
    # Place in a grid avoiding the two corners
    positions = []
    for r in range(5):
        for c in range(5):
            cx = (c + 0.5) * 0.2
            cy = (r + 0.5) * 0.2
            # Check if this position conflicts with the two large squares
            # For simplicity, just try all and keep valid ones
            positions.append([cx, cy])
    
    # Filter positions that don't overlap with large squares
    valid_positions = []
    for pos in positions:
        cx, cy = pos
        ok = True
        # Check against large square 0
        t0 = (s2 + s0) / 2
        if abs(cx - s0/2) < t0 and abs(cy - s0/2) < t0:
            ok = False
        # Check against large square 1
        t1 = (s2 + s1) / 2
        if abs(cx - (1-s1/2)) < t1 and abs(cy - (1-s1/2)) < t1:
            ok = False
        if ok:
            valid_positions.append(pos)
    
    # Take first 8 valid positions
    for pos in valid_positions[:8]:
        centers.append(pos)
        sides.append(s2)
    
    if len(centers) == 10:
        return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)
    
    # Fallback
    return _seed_grid()


def _seed_strip():
    """
    Horizontal strip: 5 squares of side 0.2 in bottom row, 5 in top row.
    Actually, let's do: 2 rows of 5, side = 0.2 (width 1.0/5=0.2, height 0.2)
    But we can make them bigger: if we do 2 rows, height = 1/2 = 0.5, width = 1/5 = 0.2, so side = 0.2.
    Better: 1 row of 10? No, too small.
    Let's do: 2 rows of 5, but make them 0.2 x 0.2.
    Or: 5 in bottom (side 0.2), 5 in top (side 0.2), but shift to allow bigger.
    """
    n = 10
    centers = []
    sides = []
    # 5 columns, 2 rows
    s = 0.2
    for r in range(2):
        for c in range(5):
            cx = (c + 0.5) * s
            cy = (r + 0.5) * s
            centers.append([cx, cy])
            sides.append(s)
    return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)


def _seed_large_small():
    """
    One very large square, surrounded by smaller ones.
    Large: 0.3 in center. Surround with 9 small ones of ~0.15.
    """
    n = 10
    centers = []
    sides = []
    
    # Center large square
    s0 = 0.3
    centers.append([0.5, 0.5])
    sides.append(s0)
    
    # 8 surrounding squares of side 0.15
    s1 = 0.15
    # Top-left, top, top-right, left, right, bottom-left, bottom, bottom-right
    offsets = [
        (0.5 - s0/2 - s1/2, 0.5 + s0/2 + s1/2),  # top-left
        (0.5, 0.5 + s0/2 + s1/2),                  # top
        (0.5 + s0/2 + s1/2, 0.5 + s0/2 + s1/2),  # top-right
        (0.5 - s0/2 - s1/2, 0.5),                  # left
        (0.5 + s0/2 + s1/2, 0.5),                  # right
        (0.5 - s0/2 - s1/2, 0.5 - s0/2 - s1/2),  # bottom-left
        (0.5, 0.5 - s0/2 - s1/2),                  # bottom
        (0.5 + s0/2 + s1/2, 0.5 - s0/2 - s1/2),  # bottom-right
    ]
    for ox, oy in offsets:
        # Clamp to [0,1]
        ox = max(s1/2, min(1-s1/2, ox))
        oy = max(s1/2, min(1-s1/2, oy))
        centers.append([ox, oy])
        sides.append(s1)
    
    # That's 9. Need 1 more.
    # Add one more small square in a corner
    s2 = 0.1
    centers.append([s2/2, s2/2])
    sides.append(s2)
    
    if len(centers) == 10:
        return np.array(centers, dtype=np.float64), np.array(sides, dtype=np.float64)
    
    return _seed_grid()


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    
    Strategy: Multi-start deterministic greedy optimization.
    1. Generate diverse initial packings (grid, heterogeneous, strip, large-small).
    2. For each, run a deterministic coordinate-descent optimizer that:
       a. Maximizes each square's side to its geometric limit.
       b. Performs position adjustments to create room for further growth.
    3. Return the best feasible packing found.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Generate diverse seeds
    seeds = [
        _seed_grid(),
        _seed_heterogeneous2(),
        _seed_strip(),
        _seed_large_small(),
    ]
    
    best_centers = None
    best_sides = None
    best_val = -float('inf')
    
    for seed_idx, (centers, sides) in enumerate(seeds):
        if time.time() > deadline - 60:
            break
        
        # Ensure seed is valid; if not, try to fix
        if not _is_valid(centers, sides, tol):
            # Shrink and reposition
            for i in range(n):
                sides[i] = 0.1
                centers[i] = [0.5 + (i % 4 - 1.5) * 0.2, 0.5 + (i // 4 - 0.5) * 0.2]
            if not _is_valid(centers, sides, tol):
                continue
        
        # Run greedy optimization
        opt_centers, opt_sides = _greedy_optimize(centers, sides, tol, max_rounds=100, time_limit=deadline - 30)
        
        # Verify
        if _is_valid(opt_centers, opt_sides, tol):
            val = _objective(opt_centers, opt_sides)
            if val > best_val:
                best_val = val
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
    
    # If no valid result, fall back to a simple valid packing
    if best_centers is None:
        centers, sides = _seed_grid()
        if not _is_valid(centers, sides, tol):
            # Last resort: tiny squares
            centers = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                                 [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]], dtype=np.float64)
            sides = np.full(10, 0.15, dtype=np.float64)
        best_centers = centers
        best_sides = sides
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
