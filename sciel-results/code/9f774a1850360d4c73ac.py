"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns 32 distinct integer pairs [a, b] on the triangular lattice
    (point a + b*omega, omega = exp(i*pi/3)) with |a|,|b| <= 64,
    mutual distance >= 1, maximizing pairs at distance exactly 1.
    Optimum is 76 (Harborth's theorem).
    """
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    N = 32

    def dist2(p, q):
        # distance squared between lattice points p=(a,b) and q=(c,d)
        # point = a + b*omega where omega = cos(60)+i sin(60) = 1/2 + i*sqrt(3)/2
        # In 2D: x = a + b/2, y = b*sqrt(3)/2
        # dist2 = (a-c)^2 + (b-d)^2 + (a-c)(b-d)
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def count_edges(points):
        """Count pairs at distance exactly 1 (dist2 == 1)."""
        cnt = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist2(points[i], points[j]) == 1:
                    cnt += 1
        return cnt

    def is_valid(points):
        """Check all pairs have dist2 >= 1 and are distinct."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
                if dist2(points[i], points[j]) < 1:
                    return False
        return True

    def hex_ring(center, radius):
        """Generate hexagonal ring of lattice points at given radius from center."""
        cx, cy = center
        if radius == 0:
            return [(cx, cy)]
        # 6 directions in axial coordinates
        dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        # Start at center + radius * first direction
        x, y = cx + radius * dirs[0][0], cy + radius * dirs[0][1]
        ring = []
        for side in range(6):
            dx, dy = dirs[side]
            for _ in range(radius):
                ring.append((x, y))
                x += dx
                y += dy
        return ring

    def all_points_in_hex(center, max_radius):
        """All points within hexagonal distance max_radius from center."""
        cx, cy = center
        pts = [(cx, cy)]
        for r in range(1, max_radius + 1):
            pts.extend(hex_ring((cx, cy), r))
        return pts

    # Strategy 1: Try a spiral/hexagonal packing directly
    # A hexagonal cluster of 32 points: rows of 6,6,6,6,6,2 or similar
    # Hexagonal numbers: 1, 7, 19, 37. So 32 fits in radius 2 (37 points) minus 5.
    # Let's try: take the hex of radius 2 (37 points) and remove 5 points
    # to maximize edges. The hex of radius 2 has 37 points.
    # Actually, let's just construct a good packing.

    # Known optimal: 32 points in a hexagonal arrangement
    # Let's use a "lozenge" or rectangular-ish shape that's known to be dense.
    # For n=32, floor(3n - sqrt(12n-3)) = floor(96 - sqrt(381)) = floor(96 - 19.52) = 76.

    # Let's try a systematic construction:
    # Place points in a hexagonal grid pattern.
    # A 6x6 hex block has 36 points. Remove 4 to get 32.
    # Or use rows: 5,6,6,6,6,3 = 32

    def make_hex_cluster(rows):
        """Build a hexagonal cluster with given row lengths (centered)."""
        pts = []
        total = sum(rows)
        if total != N:
            return None
        # Center the rows
        # Row i has rows[i] points
        # In axial coords, row i is at b = i - (len(rows)-1)//2
        # Points in row i: a from -rows[i]//2 to rows[i]//2 - 1 (if odd) or -rows[i]//2 to rows[i]//2 - 1
        mid = (len(rows) - 1) // 2
        for i, rlen in enumerate(rows):
            b = i - mid
            # a ranges to center the row
            a_start = -(rlen - 1) // 2
            for a in range(a_start, a_start + rlen):
                pts.append((a, b))
        return pts

    # Try various row configurations
    best_points = None
    best_edges = -1

    # Generate candidate row configurations that sum to 32
    # Rows should be roughly equal, within a few
    # Try all combinations of row lengths that sum to 32, with 4-8 rows
    # Each row length between 4 and 10
    def try_config(rows):
        nonlocal best_points, best_edges
        if sum(rows) != N:
            return
        pts = make_hex_cluster(rows)
        if pts is None:
            return
        if not is_valid(pts):
            return
        edges = count_edges(pts)
        if edges > best_edges:
            best_edges = edges
            best_points = pts

    # Try many row configurations
    # Number of rows from 4 to 8
    for n_rows in range(4, 9):
        # Generate all compositions of 32 into n_rows parts, each between 3 and 10
        def gen_compositions(total, parts, min_v=3, max_v=10):
            if parts == 1:
                if min_v <= total <= max_v:
                    yield [total]
                return
            for v in range(min_v, max_v + 1):
                if total - v < (parts - 1) * min_v:
                    break
                if total - v > (parts - 1) * max_v:
                    continue
                for rest in gen_compositions(total - v, parts - 1, min_v, max_v):
                    yield [v] + rest

        for rows in gen_compositions(N, n_rows):
            try_config(rows)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Also try shifted versions of the best config
    if best_points:
        # Try small shifts
        for da in range(-2, 3):
            for db in range(-2, 3):
                shifted = [(a + da, b + db) for a, b in best_points]
                if all(abs(a) <= 64 and abs(b) <= 64 for a, b in shifted):
                    if is_valid(shifted):
                        edges = count_edges(shifted)
                        if edges > best_edges:
                            best_edges = edges
                            best_points = shifted

    # If we haven't found a good solution, use a known good one
    # Fallback: a 16x2 strip has 16*2 - 16 = 15 horizontal + 16 vertical... 
    # Actually 16x2: horizontal edges = 15*2 = 30, vertical = 16, diagonal = 15+15=30? 
    # Let's just use the hex cluster if found, else the strip.
    if best_points is None or best_edges < 70:
        # Use a better fallback: hexagonal arrangement
        # 32 points: try rows [6,6,6,6,6,2] or [5,6,6,6,6,3]
        fallback_configs = [
            [6, 6, 6, 6, 6, 2],
            [5, 6, 6, 6, 6, 3],
            [4, 6, 6, 6, 6, 4],
            [5, 5, 6, 6, 6, 4],
            [6, 5, 6, 6, 5, 4],
        ]
        for rows in fallback_configs:
            if sum(rows) != N:
                continue
            pts = make_hex_cluster(rows)
            if pts and is_valid(pts):
                edges = count_edges(pts)
                if edges > best_edges:
                    best_edges = edges
                    best_points = pts

    # Final fallback: the original strip
    if best_points is None:
        best_points = [[a, b] for a in range(16) for b in range(2)]

    # Convert to list of lists
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
