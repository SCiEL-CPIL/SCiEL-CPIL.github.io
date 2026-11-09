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


def _max_grow(centers, sides, i, tol=1e-7):
    """
    Compute the maximum additional side length square i can grow by,
    keeping it centered (i.e., growing symmetrically about its center).
    Returns (max_delta, limiting_constraint).
    """
    cx, cy = centers[i]
    s = sides[i]
    # Wall constraints: half-side can grow to at most min(cx, 1-cx) in x, min(cy, 1-cy) in y
    max_half_x = min(cx, 1.0 - cx)
    max_half_y = min(cy, 1.0 - cy)
    max_half = min(max_half_x, max_half_y)
    max_from_walls = max_half * 2 - s  # total side from walls
    
    # Neighbor constraints: for each other square j, find max growth
    max_from_neighbors = max_from_walls
    for j in range(len(sides)):
        if j == i:
            continue
        dx = abs(centers[i, 0] - centers[j, 0])
        dy = abs(centers[i, 1] - centers[j, 1])
        # Current gap in x and y
        # Overlap condition: |cx_i - cx_j| < (s_i + s_j)/2
        # For growth by delta (s_i -> s_i + delta), need |dx| >= (s_i + delta + s_j)/2
        # So delta <= 2*|dx| - s_i - s_j  (if |dx| is the binding axis)
        # Similarly for y
        # The constraint is: max(|dx|, |dy|) >= (s_i + delta + s_j)/2
        # Actually the overlap condition is BOTH |dx| < t AND |dy| < t
        # So to avoid overlap, we need |dx| >= t OR |dy| >= t where t = (s_i+delta+s_j)/2
        # The binding constraint is the minimum of the two gaps
        # gap_x = |dx| - s_j/2  (distance from edge of i to edge of j in x, before growth)
        # gap_y = |dy| - s_j/2
        # After growing i by delta: need gap_x - delta/2 >= 0 OR gap_y - delta/2 >= 0
        # So delta <= 2*gap_x OR delta <= 2*gap_y
        # The constraint is: delta <= 2 * max(gap_x, gap_y)  ... no wait
        # Actually: overlap if (|dx| - (s_i+delta)/2 - s_j/2 < 0) AND (|dy| - (s_i+delta)/2 - s_j/2 < 0)
        # = (|dx| - s_i/2 - delta/2 - s_j/2 < 0) AND (|dy| - s_i/2 - delta/2 - s_j/2 < 0)
        # = (|dx| - (s_i+s_j)/2 < delta/2) AND (|dy| - (s_i+s_j)/2 < delta/2)
        # To avoid: |dx| - (s_i+s_j)/2 >= delta/2 OR |dy| - (s_i+s_j)/2 >= delta/2
        # So delta <= 2*(|dx| - (s_i+s_j)/2) OR delta <= 2*(|dy| - (s_i+s_j)/2)
        # The max allowed delta from this neighbor is 2 * max(|dx| - (s_i+s_j)/2, |dy| - (s_i+s_j)/2)
        # But we need delta such that the OR holds, so delta <= 2*max(gap_x, gap_y) where gap = |d| - (s_i+s_j)/2
        gap_x = dx - (s + sides[j]) / 2
        gap_y = dy - (s + sides[j]) / 2
        # If both gaps are negative, we're already overlapping (shouldn't happen if valid)
        max_delta_j = 2.0 * max(gap_x, gap_y)
        if max_delta_j < max_from_neighbors:
            max_from_neighbors = max_delta_j
    
    max_delta = min(max_from_walls, max_from_neighbors)
    return max(0.0, max_delta)


def _greedy_enlarge(centers, sides, tol=1e-7, max_rounds=200, time_limit=None, start_time=None):
    """
    Greedy iterative enlargement: repeatedly find the square that can grow the most,
    grow it by a fraction of its max, and repeat.
    """
    centers = centers.copy()
    sides = sides.copy()
    n = len(sides)
    
    for rnd in range(max_rounds):
        if time_limit is not None:
            if start_time is not None and time.time() - start_time > time_limit:
                break
        
        # Find the square with maximum possible growth
        best_i = -1
        best_delta = 0.0
        for i in range(n):
            d = _max_grow(centers, sides, i, tol)
            if d > best_delta:
                best_delta = d
                best_i = i
        
        if best_i < 0 or best_delta < 1e-10:
            break
        
        # Grow by 95% of max (to avoid numerical issues)
        growth = best_delta * 0.95
        sides[best_i] += growth
        
        # Verify validity
        if not _is_valid(centers, sides, tol):
            # Try smaller growth
            growth = best_delta * 0.5
            sides[best_i] += growth
            if not _is_valid(centers, sides, tol):
                growth = best_delta * 0.25
                sides[best_i] += growth
                if not _is_valid(centers, sides, tol):
                    sides[best_i] -= growth
                    break
        else:
            sides[best_i] += 0  # already added
    
    return centers, sides


def _seed_heterogeneous():
    """
    Create a heterogeneous initial packing:
    - 1 large square (side ~0.4) in top-left
    - Remaining 9 squares fill the L-shaped region
    
    Layout concept:
    Large square at top-left: center (0.2, 0.8), side 0.4
    This leaves:
    - Right strip: x in [0.4, 1], y in [0, 1]  -> width 0.6, height 1.0
    - Bottom-left strip: x in [0, 0.4], y in [0, 0.6] -> width 0.4, height 0.6
    
    Fill right strip with 2 columns of squares
    Fill bottom-left with 1 column
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # Square 0: large, top-left
    sides[0] = 0.4
    centers[0] = [0.2, 0.8]
    
    # Right region: x in [0.4, 1.0], y in [0, 1.0]
    # Use 2 columns x 4 rows = 8 squares, but we need 9 more
    # Let's do 3 columns x 3 rows in the right region = 9 squares
    # Right region: width 0.6, height 1.0
    # 3 cols: each 0.2 wide, 3 rows: each ~0.333 tall
    # But 0.2 x 0.333 isn't square. Use side = 0.2 for all in right region
    # 3 cols x 5 rows = 15, too many. 
    # Let's use: right region 3 cols x 3 rows with side 0.2, centered vertically
    # That's 9 squares of side 0.2 in the right region
    
    # Actually, let me design more carefully:
    # Large square: side 0.4 at top-left [0,0.4]x[0.6,1.0]
    # Below it: [0,0.4]x[0,0.6] - can fit 2 squares of side 0.3 stacked? No, 0.4 wide.
    #   2 squares of side 0.2 side by side: [0,0.2]x[0,0.2] and [0.2,0.4]x[0,0.2]
    #   2 more of side 0.2: [0,0.2]x[0.2,0.4] and [0.2,0.4]x[0.2,0.4]
    #   1 more of side 0.2: [0,0.2]x[0.4,0.6] -- but that's 5 in bottom-left
    # Right region [0.4,1.0]x[0,1.0]: width 0.6, height 1.0
    #   3 cols of side 0.2, 5 rows of side 0.2 = 15, too many
    #   3 cols of side 0.2, 3 rows = 9 squares, using height 0.6, centered in 1.0
    #   Or 2 cols of side 0.3, 3 rows of side 0.3 = 6 squares using 0.6x0.9
    
    # Let me just use a simpler approach:
    # 1 large (0.4) + 9 small (0.2)
    # Large at [0, 0.4] x [0.6, 1.0], center (0.2, 0.8)
    # 9 small of side 0.2:
    #   In [0, 0.4] x [0, 0.6]: 2 cols x 3 rows = 6 squares
    #   In [0.4, 1.0] x [0, 1.0]: need 3 more. 1 col x 3 rows of side 0.2 at x=[0.4,0.6]
    #   Wait, that leaves [0.6, 1.0] x [0, 1.0] empty.
    
    # Better: 
    # Large: side 0.4, center (0.2, 0.8) -> [0,0.4]x[0.6,1.0]
    # Bottom-left [0,0.4]x[0,0.6]: 2x3 grid of side 0.2 = 6 squares
    #   centers: x in {0.1, 0.3}, y in {0.1, 0.3, 0.5}
    # Right [0.4,1.0]x[0,1.0]: 3 squares of side 0.2 in a vertical line at x=0.5
    #   centers: (0.5, 0.1), (0.5, 0.3), (0.5, 0.5)
    #   But that only uses [0.4,0.6]x[0,0.6]. The rest [0.6,1.0] is empty.
    #   We could make the 3 right squares larger.
    
    # Let's try: 3 squares of side 0.3 in the right region
    # [0.4, 1.0] x [0, 1.0]: width 0.6, height 1.0
    # 2 cols of side 0.3: x in {0.55, 0.85}
    # 3 rows of side 0.3: y in {0.15, 0.45, 0.75}
    # That's 6 squares. We need 3 more.
    # 
    # Total so far: 1 large + 6 bottom-left + 6 right = 13. Too many.
    # We need exactly 10 total.
    # 1 large + 9 small.
    
    # Let me restart with a cleaner layout:
    # Square 0: side 0.5, center (0.25, 0.75) -> [0, 0.5] x [0.5, 1.0]
    # Remaining: [0.5, 1.0] x [0, 1.0] (right, 0.5x1.0) and [0, 0.5] x [0, 0.5] (bottom-left, 0.5x0.5)
    # Bottom-left: 2x2 grid of side 0.25 = 4 squares
    #   centers: (0.125, 0.125), (0.375, 0.125), (0.125, 0.375), (0.375, 0.375)
    # Right: 2 cols x 2 rows of side 0.25 = 4 squares
    #   centers: (0.625, 0.125), (0.875, 0.125), (0.625, 0.375), (0.875, 0.375)
    # That's 1 + 4 + 4 = 9. Need 1 more.
    # 
    # Alternative: 
    # Square 0: side 0.5, center (0.25, 0.75) -> [0, 0.5] x [0.5, 1.0]
    # Bottom-left [0, 0.5] x [0, 0.5]: 1 square of side 0.5? No, that fills it.
    #   Or 2 squares of side 0.25 stacked + 2 side by side... 
    #
    # Let me just go with:
    # 1 square of side 0.5 at top-left
    # 4 squares of side 0.25 in bottom-left (2x2)
    # 5 squares in the right half [0.5, 1.0] x [0, 1.0]
    #   5 squares of side 0.2: 
    #   col 1 (x=0.6): 5 rows? No, height is 1.0, 5 rows of 0.2 = 1.0. 
    #   centers: (0.6, 0.1), (0.6, 0.3), (0.6, 0.5), (0.6, 0.7), (0.6, 0.9)
    #   But these have side 0.2, so [0.5, 0.7] x ... 
    #   That works! All within [0.5, 1.0] x [0, 1.0]
    
    # Final layout:
    # 0: side 0.5, center (0.25, 0.75)
    # 1-4: side 0.25, centers (0.125,0.125), (0.375,0.125), (0.125,0.375), (0.375,0.375)
    # 5-9: side 0.2, centers (0.6, 0.1), (0.6, 0.3), (0.6, 0.5), (0.6, 0.7), (0.6, 0.9)
    
    sides[0] = 0.5
    centers[0] = [0.25, 0.75]
    
    bl = [(0.125, 0.125), (0.375, 0.125), (0.125, 0.375), (0.375, 0.375)]
    for k, (x, y) in enumerate(bl):
        sides[1 + k] = 0.25
        centers[1 + k] = [x, y]
    
    rt = [(0.6, 0.1), (0.6, 0.3), (0.6, 0.5), (0.6, 0.7), (0.6, 0.9)]
    for k, (x, y) in enumerate(rt):
        sides[5 + k] = 0.2
        centers[5 + k] = [x, y]
    
    return centers, sides


def _seed_heterogeneous2():
    """
    Alternative: 2 large squares + 8 small
    Large 1: side 0.4 at [0, 0.4] x [0.6, 1.0], center (0.2, 0.8)
    Large 2: side 0.4 at [0.6, 1.0] x [0.6, 1.0], center (0.8, 0.8)
    Gap between them: [0.4, 0.6] x [0.6, 1.0] - width 0.2
    Bottom: [0, 1.0] x [0, 0.6] - full width, height 0.6
    8 small squares of side 0.2 in bottom: 5 cols x ... no, 4 cols x 2 rows = 8
    centers: x in {0.1, 0.3, 0.5, 0.7}, y in {0.1, 0.3}
    Wait, 0.5 is in the gap. Let me use x in {0.1, 0.3, 0.7, 0.9} for 4 cols, 2 rows = 8
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    sides[0] = 0.4
    centers[0] = [0.2, 0.8]
    
    sides[1] = 0.4
    centers[1] = [0.8, 0.8]
    
    # 8 small in bottom [0,1]x[0,0.6], side 0.2
    # 4 cols x 2 rows
    xs = [0.1, 0.3, 0.7, 0.9]
    ys = [0.1, 0.3]
    idx = 2
    for x in xs:
        for y in ys:
            sides[idx] = 0.2
            centers[idx] = [x, y]
            idx += 1
    
    return centers, sides


def _seed_heterogeneous3():
    """
    3 medium squares + 7 small
    Top row: 3 squares of side 0.333 at y=0.833, x=0.167, 0.5, 0.833
    Bottom: 7 squares of side 0.2 in [0,1]x[0,0.667]
    4 cols x 2 rows = 8, use 7
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # 3 top squares
    for k, x in enumerate([0.167, 0.5, 0.833]):
        sides[k] = 0.333
        centers[k] = [x, 0.833]
    
    # 7 bottom squares of side 0.2
    # 4 cols x 2 rows, skip last
    xs = [0.1, 0.3, 0.7, 0.9]
    ys = [0.1, 0.3]
    idx = 3
    for x in xs:
        for y in ys:
            if idx >= 10:
                break
            sides[idx] = 0.2
            centers[idx] = [x, y]
            idx += 1
        if idx >= 10:
            break
    
    return centers, sides


def _local_refine(centers, sides, tol=1e-7, time_budget=120.0, start_time=None):
    """
    Local refinement: for each square, try small moves in 8 directions + size changes.
    Accept if valid and improves objective.
    """
    centers = centers.copy()
    sides = sides.copy()
    n = len(sides)
    current_val = _objective(centers, sides)
    
    if start_time is None:
        start_time = time.time()
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.001]
    
    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break
        
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            for i in range(n):
                if time.time() - start_time > time_budget:
                    break
                
                # Try position moves
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_c = centers.copy()
                        new_c[i, 0] += dx
                        new_c[i, 1] += dy
                        # Clamp
                        new_c[i, 0] = max(sides[i] / 2, min(1 - sides[i] / 2, new_c[i, 0]))
                        new_c[i, 1] = max(sides[i] / 2, min(1 - sides[i] / 2, new_c[i, 1]))
                        
                        if _is_valid(new_c, sides, tol):
                            new_val = _objective(new_c, sides)
                            if new_val > current_val + 1e-12:
                                centers = new_c
                                current_val = new_val
                                improved = True
                
                # Try size increase
                for ds in [step, step * 0.5]:
                    new_s = sides.copy()
                    new_s[i] += ds
                    if new_s[i] > 1.0:
                        continue
                    # Check if square still fits in unit square
                    if centers[i, 0] - new_s[i] / 2 < -tol or centers[i, 0] + new_s[i] / 2 > 1 + tol:
                        continue
                    if centers[i, 1] - new_s[i] / 2 < -tol or centers[i, 1] + new_s[i] / 2 > 1 + tol:
                        continue
                    if _is_valid(centers, new_s, tol):
                        new_val = _objective(centers, new_s)
                        if new_val > current_val + 1e-12:
                            sides = new_s
                            current_val = new_val
                            improved = True
    
    return centers, sides, current_val


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place 10 squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths.

    Strategy: Structured heterogeneous packings + greedy enlargement + local refinement.
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 630s limit
    
    # Generate multiple heterogeneous seeds
    seeds = []
    seeds.append(_seed_heterogeneous())
    seeds.append(_seed_heterogeneous2())
    seeds.append(_seed_heterogeneous3())
    
    # Also try a uniform grid as baseline
    k = 3
    side = 1.0 / k
    idx = np.arange(10)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    grid_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    grid_sides = np.full(10, side, dtype=np.float64)
    seeds.append((grid_centers, grid_sides))
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    for seed_idx, (sc, ss) in enumerate(seeds):
        if time.time() > deadline:
            break
        
        # Validate seed
        if not _is_valid(sc, ss):
            continue
        
        val = _objective(sc, ss)
        
        # Greedy enlargement
        ec, es = _greedy_enlarge(sc, ss, time_limit=60.0, start_time=start_time)
        if _is_valid(ec, es):
            eval_ = _objective(ec, es)
            if eval_ > val:
                sc, ss = ec, es
                val = eval_
        
        # Local refinement
        rc, rs, rval = _local_refine(sc, ss, time_budget=90.0, start_time=start_time)
        if _is_valid(rc, rs):
            if rval > val:
                sc, ss = rc, rs
                val = rval
        
        if val > best_val:
            best_val = val
            best_centers = sc.copy()
            best_sides = ss.copy()
    
    # If no valid seed found (shouldn't happen), fallback
    if best_centers is None:
        k = 3
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
