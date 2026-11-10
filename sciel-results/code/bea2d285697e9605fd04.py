# EVOLVE-BLOCK-START
import numpy as np
import time

_TOL = 1e-7
_N = 10


def _is_valid(centers, sides):
    if centers is None or sides is None:
        return False
    try:
        c = np.asarray(centers, dtype=np.float64)
        s = np.asarray(sides, dtype=np.float64)
    except Exception:
        return False
    if c.shape != (10, 2) or s.shape != (10,):
        return False
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(s))):
        return False
    if np.any(s <= 0.0):
        return False
    if np.any(c - s / 2.0 < -_TOL) or np.any(c + s / 2.0 > 1.0 + _TOL):
        return False
    n = 10
    for i in range(n):
        for j in range(i + 1, n):
            t = 0.5 * (s[i] + s[j]) - _TOL
            if abs(c[i, 0] - c[j, 0]) < t and abs(c[i, 1] - c[j, 1]) < t:
                return False
    return True


def _objective(centers, sides):
    return float(np.sum(sides))


def _grid_seed():
    k = 3
    side = 1.0 / k
    idx = np.arange(_N)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    return np.stack([cx, cy], axis=1).astype(np.float64), np.full(_N, side, dtype=np.float64)


def _hetero_seed():
    # (center, side) for 10 squares: 1 large, 1 medium, 8 small in remaining space
    squares = [
        ((0.25, 0.25), 0.50),   # 0: large bottom-left
        ((0.75, 0.75), 0.40),   # 1: medium top-right
        ((0.75, 0.15), 0.20),   # 2: bottom-right small
        ((0.15, 0.75), 0.20),   # 3: top-left small
        ((0.50, 0.50), 0.10),   # 4: center tiny
        ((0.50, 0.10), 0.10),   # 5
        ((0.10, 0.50), 0.10),   # 6
        ((0.90, 0.50), 0.10),   # 7
        ((0.50, 0.90), 0.10),   # 8
        ((0.30, 0.90), 0.10),   # 9
    ]
    centers = np.array([sq[0] for sq in squares], dtype=np.float64)
    sides = np.array([sq[1] for sq in squares], dtype=np.float64)
    return centers, sides


def _shelf_seed():
    # Shelf layout: 2 rows.
    # Row 0 (bottom, y in [0, 0.45]): 3 squares of side 0.45 -> x: [0,0.45],[0.45,0.90],[0.90,1.35]? no, must fit in [0,1].
    # Let's do: Row 0: two squares of side 0.5 -> x: [0,0.5], [0.5,1.0]. y: [0,0.5].
    # Row 1 (top, y in [0.5, 1.0]): 8 squares? Need 8 more.
    # Actually, let's do a 2x5 grid of side 0.2 -> 10 squares. Sum = 2.0. Worse than grid.
    # Let's try: 1 big (0.5) bottom-left.
    # Remaining: right strip (0.5 x 1) and top strip (1 x 0.5) minus overlap (0.5x0.5).
    # Right strip: 2 squares of side 0.5? No, 0.5x0.5 fits in 0.5x1. So two 0.5 squares in right strip?
    # If we put one 0.5 at (0.75, 0.25) and one at (0.75, 0.75).
    # Top-left area: [0, 0.5] x [0.5, 1.0].
    # We have 1 (big) + 2 (right) = 3 squares. Need 7 more.
    # Top-left 0.5x0.5 area can hold 4 squares of side 0.25.
    # That's 7 total. Need 3 more.
    # This is getting complicated to guarantee validity without search.
    # Let's stick to a known valid heterogeneous seed and let the shelf search improve it.
    # Actually, the shelf search below will generate its own valid layouts.
    # Let's just return the grid seed for now, the shelf search will do the work.
    return _grid_seed()


def _generate_shelf_layouts():
    """
    Generate a set of valid shelf-like layouts.
    Strategy: Partition [0,1]x[0,1] into rectangles (shelves), and fill them with squares.
    We'll try a few specific partitioning patterns.
    """
    layouts = []
    
    # Pattern 1: 2 horizontal shelves
    # Shelf 1: y in [0, h1], Shelf 2: y in [h1, 1]
    # Fill Shelf 1 with k1 squares, Shelf 2 with k2 squares. k1+k2=10.
    # For simplicity, let's try uniform squares in shelves.
    # If Shelf 1 has height h1, max square side is h1.
    # If we place k1 squares of side s1 = h1 in Shelf 1, we need k1 * h1 <= 1.
    # Similarly for Shelf 2.
    
    # Let's try a specific "corner" pattern which is often optimal for small n.
    # One large square in a corner, and smaller ones filling the rest.
    
    # Pattern A: One 0.5 square at (0.25, 0.25).
    # Remaining L-shape: [0.5,1]x[0,1] and [0,0.5]x[0.5,1].
    # In [0.5,1]x[0,1] (0.5 x 1), we can fit two 0.5 squares? No, 0.5+0.5=1.0 width, but height is 1.0.
    # So two 0.5 squares: (0.75, 0.25) and (0.75, 0.75).
    # Now we have 3 squares of side 0.5. Sum = 1.5.
    # Remaining area: [0, 0.5] x [0.5, 1.0]. Size 0.5 x 0.5.
    # We need 7 more squares.
    # In 0.5x0.5, we can fit 4 squares of 0.25.
    # Centers: (0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875).
    # That's 4 squares. Total 7 squares.
    # We need 3 more.
    # This pattern is stuck at 7 squares if we stick to grid.
    # But we can resize!
    # The 4 small squares in 0.5x0.5 can be enlarged if we rearrange? No, they are touching.
    # What if we don't use 0.5 for the big ones?
    
    # Let's try a different pattern:
    # 2 large squares of side 0.4.
    # (0.2, 0.2) and (0.8, 0.8).
    # Sum = 0.8.
    # Remaining space is complex.
    
    # Actually, the most robust "shelf" approach for 10 squares:
    # Try to find a layout that is a refinement of a coarse grid.
    # Let's just use the heterogeneous seed and the grid seed as the starting points for the local search.
    # The "shelf" idea in the prompt's inspiration 1 was about SA.
    # My "structural change" is to use a **combinatorial layout search** that explicitly constructs valid packings from a grammar,
    # rather than starting from a continuous point and moving it.
    
    # Let's implement a simple recursive subdivision search.
    # State: list of rectangles.
    # Action: Split a rectangle into two (horizontal or vertical), or place a square in it.
    # This is a tree search.
    
    # To keep it fast and simple, I will hardcode a few high-quality "templates" 
    # that are known to be good packings for 10 squares, 
    # and then use the local search to refine them.
    
    # Template 1: The "1 large, 1 medium, 8 small" from _hetero_seed.
    # Template 2: A 3x3 grid with one cell merged? No, must be squares.
    # Template 3: 5 squares of side 0.2 in a row, and 5 in another? Sum = 2.0. Bad.
    
    # Let's try to generate a better template programmatically.
    # We want to maximize sum of sides.
    # Ideally, we want squares as large as possible.
    # The theoretical max is related to packing density.
    
    # Let's try this layout:
    # 2 squares of side 0.5: (0.25, 0.25) and (0.75, 0.75).
    # 2 squares of side 0.5: (0.75, 0.25) and (0.25, 0.75).
    # That's 4 squares of 0.5. Sum = 2.0.
    # Remaining: 4 corner regions of 0.5x0.5? No, they are covered.
    # Wait, (0.25,0.25) covers [0,0.5]x[0,0.5].
    # (0.75,0.75) covers [0.5,1]x[0.5,1].
    # (0.75,0.25) covers [0.5,1]x[0,0.5].
    # (0.25,0.75) covers [0,0.5]x[0.5,1].
    # This covers the whole square with 4 squares of side 0.5.
    # We need 6 more. We can subdivide each 0.5x0.5 square into 4 squares of 0.25.
    # If we subdivide all 4, we have 16 squares of 0.25.
    # We need 10.
    # So we can keep 2 of the original 0.5 squares, and subdivide the other 2 into 4 each.
    # Total squares: 2 + 4 + 4 = 10.
    # Sides: 0.5, 0.5, and eight 0.25s.
    # Sum = 0.5 + 0.5 + 8 * 0.25 = 1.0 + 2.0 = 3.0.
    # This is valid!
    # Centers:
    # Sq 0: (0.25, 0.25), s=0.5
    # Sq 1: (0.75, 0.75), s=0.5
    # Subdivide [0.5,1]x[0,0.5] into 4:
    #   (0.625, 0.125), (0.875, 0.125), (0.625, 0.375), (0.875, 0.375), s=0.25
    # Subdivide [0,0.5]x[0.5,1] into 4:
    #   (0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875), s=0.25
    
    centers = np.array([
        [0.25, 0.25],
        [0.75, 0.75],
        [0.625, 0.125],
        [0.875, 0.125],
        [0.625, 0.375],
        [0.875, 0.375],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.125, 0.875],
        [0.375, 0.875]
    ], dtype=np.float64)
    sides = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ], dtype=np.float64)
    
    if _is_valid(centers, sides):
        layouts.append((centers, sides))
        
    # Try another variant: 1 large, 3 medium, 6 small?
    # Large 0.5 at (0.25, 0.25).
    # Right strip [0.5,1]x[0,1]:
    #   Top: 0.5x0.5 at (0.75, 0.75).
    #   Bottom: 0.5x0.5 at (0.75, 0.25).
    # That's 3 large.
    # Top-left [0,0.5]x[0.5,1]:
    #   4 squares of 0.25.
    # Total: 3 large + 4 small = 7 squares. Need 3 more.
    # We can subdivide one of the large ones?
    # If we subdivide the bottom-right large (0.75, 0.25) into 4 smalls.
    # Total: 2 large + 8 small = 10 squares.
    # This is the same as the previous one, just rotated.
    
    # What if we use 4 large and 6 small?
    # 4 large of 0.5 covers the whole square.
    # We need 6 small.
    # We can subdivide 1.5 large squares? No, must be integer.
    # So we can subdivide 2 large squares into 4 smalls each -> 8 smalls.
    # Total 2 large + 8 small = 10. (This is the layout above).
    # Or subdivide 1 large square into 4 smalls, and keep 3 large.
    # Total 3 large + 4 small = 7. Need 3 more.
    # We can subdivide one of the remaining large squares into 4 smalls?
    # Then 2 large + 8 small = 10. Same.
    
    # It seems 2 large + 8 small is the only integer subdivision of the 4-large grid.
    # Sum = 3.0.
    # The grid baseline is 10 * (1/3) = 3.333.
    # So this template is WORSE than the grid baseline!
    # Why? Because 1/3 > 0.25 and 0.5.
    # 10 * 0.333 = 3.333.
    # 2 * 0.5 + 8 * 0.25 = 3.0.
    # So the grid is better than this simple subdivision.
    
    # We need a smarter packing.
    # The grid of 1/3 is 3x3.
    # Can we do better than 1/3?
    # If we use 2x5 grid, side = 0.2. Sum = 2.0.
    # If we use 5x2 grid, side = 0.2.
    # If we use 1x10 grid, side = 0.1.
    
    # The optimal packing for 10 squares is known to be better than 3.333.
    # Actually, the problem is to MAXIMIZE sum of side lengths.
    # This is different from maximizing the minimum side length (which is the usual "squares in a square" problem).
    # Here, we want some squares to be very large, and others to be small.
    # The sum is linear in side lengths.
    # So we want to make the large squares as large as possible.
    
    # Let's try: 1 square of side 0.5.
    # Remaining area: L-shape.
    # In the L-shape, we want to fit 9 squares.
    # The L-shape consists of a 0.5x1 rectangle and a 0.5x0.5 square.
    # In the 0.5x1 rectangle, we can fit 2 squares of 0.5.
    # So we have 3 squares of 0.5.
    # Remaining: 0.5x0.5 square.
    # We need 7 more squares in 0.5x0.5.
    # Max side for 7 squares in 0.5x0.5?
    # If we use a 3x3 grid in 0.5x0.5, side = 0.5/3 = 0.166.
    # Sum = 3 * 0.5 + 7 * 0.166 = 1.5 + 1.166 = 2.666.
    # This is worse than 3.333.
    
    # What if we use 1 square of side 0.4?
    # Remaining area is larger.
    # Let's try to optimize the "1 large, rest small" strategy.
    # Let the large square have side L.
    # The remaining area is roughly (1-L)*1 + L*(1-L) = 1 - L^2.
    # We need to fit 9 squares in this area.
    # If we pack them in a grid, the side s is approx sqrt((1-L^2)/9).
    # Sum = L + 9 * s = L + 3 * sqrt(1-L^2).
    # Let's maximize f(L) = L + 3 * sqrt(1-L^2).
    # f'(L) = 1 - 3L/sqrt(1-L^2) = 0 => sqrt(1-L^2) = 3L => 1-L^2 = 9L^2 => 10L^2 = 1 => L = 1/sqrt(10) approx 0.316.
    # s = 3 * 0.316 / sqrt(10) = 0.3.
    # Sum = 0.316 + 9 * 0.3 = 0.316 + 2.7 = 3.016.
    # Still worse than 3.333.
    
    # It seems the uniform grid of 1/3 is quite good for this specific objective.
    # But the inspiration says "Heterogeneous packings... can exceed the uniform 1/3 grid sum of 3.33".
    # How?
    # Maybe by using non-grid arrangements that allow larger squares.
    # For example, if we place a 0.5 square, we lose a lot of area for the others.
    # But if we place a 0.4 square, we might fit the others more efficiently.
    
    # Let's try a specific layout:
    # 2 squares of side 0.4.
    # (0.2, 0.2) and (0.8, 0.8).
    # Sum = 0.8.
    # Remaining:
    #   [0.4, 0.6] x [0, 1] (vertical strip)
    #   [0, 1] x [0.4, 0.6] (horizontal strip)
    #   Intersection [0.4, 0.6] x [0.4, 0.6] (0.2x0.2 square)
    #   [0, 0.4] x [0.6, 1] (0.4x0.4 square)
    #   [0.6, 1] x [0, 0.4] (0.4x0.4 square)
    #   [0, 0.4] x [0, 0.4] is covered by first square? No, first is [0,0.4]x[0,0.4]. Yes.
    #   [0.6, 1] x [0.6, 1] is covered by second square? Yes.
    
    # So remaining areas:
    # 1. Vertical strip: [0.4, 0.6] x [0, 0.4] and [0.4, 0.6] x [0.6, 1].
    #    Actually, the vertical strip is [0.4, 0.6] x [0, 1] minus the two 0.4 squares.
    #    The two 0.4 squares are at bottom-left and top-right.
    #    So the vertical strip is [0.4, 0.6] x [0, 0.4] (below the top square) and [0.4, 0.6] x [0.6, 1] (above the bottom square)?
    #    No, the bottom square is [0,0.4]x[0,0.4].
    #    So [0.4, 0.6] x [0, 0.4] is free.
    #    The top square is [0.6,1]x[0.6,1].
    #    So [0.4, 0.6] x [0.6, 1] is free.
    #    And [0.4, 0.6] x [0.4, 0.6] is free.
    #    So the whole strip [0.4, 0.6] x [0, 1] is free?
    #    Wait, the bottom square is [0,0.4]x[0,0.4]. It does not touch x=0.4.
    #    The top square is [0.6,1]x[0.6,1]. It does not touch x=0.4.
    #    So the strip [0.4, 0.6] x [0, 1] is completely free.
    #    Width 0.2, Height 1.
    #    We can fit 5 squares of side 0.2 in this strip.
    #    Centers: (0.5, 0.1), (0.5, 0.3), (0.5, 0.5), (0.5, 0.7), (0.5, 0.9).
    #    Sum = 5 * 0.2 = 1.0.
    
    # 2. Horizontal strip: [0, 0.4] x [0.4, 0.6] and [0.6, 1] x [0.4, 0.6].
    #    Wait, the horizontal strip is [0, 1] x [0.4, 0.6] minus the two 0.4 squares.
    #    The bottom square is [0,0.4]x[0,0.4]. It does not touch y=0.4.
    #    The top square is [0.6,1]x[0.6,1]. It does not touch y=0.4.
    #    So the strip [0, 1] x [0.4, 0.6] is completely free?
    #    No, the vertical strip [0.4, 0.6] x [0, 1] intersects it.
    #    We already counted the intersection [0.4, 0.6] x [0.4, 0.6] in the vertical strip.
    #    So the remaining horizontal parts are:
    #    [0, 0.4] x [0.4, 0.6] (0.4 x 0.2)
    #    [0.6, 1] x [0.4, 0.6] (0.4 x 0.2)
    #    In each 0.4x0.2 rectangle, we can fit 2 squares of side 0.2.
    #    So 4 squares of side 0.2.
    #    Sum = 4 * 0.2 = 0.8.
    
    # Total squares: 2 (large) + 5 (vertical) + 4 (horizontal) = 11 squares.
    # We need 10.
    # We can remove one of the small squares.
    # Sum = 0.8 + 1.0 + 0.8 - 0.2 = 2.4.
    # Still worse than 3.333.
    
    # It seems hard to beat the 1/3 grid with simple layouts.
    # But the local search might find a better one.
    # Let's just use the grid seed and the heterogeneous seed, and run the local search.
    
    return layouts


def _greedy_enlarge(centers, sides, max_iters=1000):
    """
    Try to enlarge each square as much as possible, one by one,
    while maintaining validity.
    """
    c = centers.copy()
    s = sides.copy()
    
    for _ in range(max_iters):
        improved = False
        for i in range(_N):
            # Try to increase s[i] by a small amount
            # Max possible side is limited by boundaries and other squares.
            # Let's just try a small step and check validity.
            # To be efficient, we can compute the max possible side for square i
            # given the positions of others.
            
            # Max side based on boundaries:
            # s_i <= 2 * min(cx_i, 1-cx_i, cy_i, 1-cy_i)
            max_s_bound = 2.0 * min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
            
            # Max side based on other squares:
            # For each j, if they overlap in x, they must not overlap in y, and vice versa.
            # This is complex. Let's just use a binary search or small steps.
            # Given the time limit, small steps are safer.
            
            step = 0.01
            while s[i] + step <= max_s_bound:
                s_new = s.copy()
                s_new[i] += step
                if _is_valid(c, s_new):
                    s[i] += step
                    improved = True
                else:
                    break
        if not improved:
            break
    return c, s


def _local_refine(centers, sides, time_budget=10.0):
    """
    Simple local search: move centers slightly, and resize.
    """
    c = centers.copy()
    s = sides.copy()
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        improved = False
        # Try moving each square slightly
        for i in range(_N):
            for dx in [-0.01, 0.01]:
                for dy in [-0.01, 0.01]:
                    c_new = c.copy()
                    c_new[i, 0] += dx
                    c_new[i, 1] += dy
                    if _is_valid(c_new, s):
                        # Check if objective improved (it won't, since sides are same)
                        # But it might allow future enlargement.
                        # So we accept if it's valid.
                        c = c_new
                        improved = True
        # Try enlarging each square
        for i in range(_N):
            max_s_bound = 2.0 * min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
            if s[i] < max_s_bound:
                step = 0.01
                while s[i] + step <= max_s_bound:
                    s_new = s.copy()
                    s_new[i] += step
                    if _is_valid(c, s_new):
                        s[i] += step
                        improved = True
                    else:
                        break
        if not improved:
            break
    return c, s


def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Generate candidate seeds
    seeds = []
    c_grid, s_grid = _grid_seed()
    seeds.append((c_grid, s_grid))
    
    c_het, s_het = _hetero_seed()
    if _is_valid(c_het, s_het):
        seeds.append((c_het, s_het))
    
    for c_shelf, s_shelf in _generate_shelf_layouts():
        seeds.append((c_shelf, s_shelf))
    
    best_centers = None
    best_sides = None
    best_score = -1.0
    
    for c, s in seeds:
        if time.time() > deadline - 60.0:
            break
        # Refine
        c_ref, s_ref = _local_refine(c, s, time_budget=30.0)
        score = _objective(c_ref, s_ref)
        if score > best_score:
            best_score = score
            best_centers = c_ref
            best_sides = s_ref
    
    if best_centers is None:
        best_centers, best_sides = _grid_seed()
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
