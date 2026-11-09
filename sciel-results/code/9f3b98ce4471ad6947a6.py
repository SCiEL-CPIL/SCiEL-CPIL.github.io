"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import Counter

def _score(points):
    """Compute the maximum multiplicity of any single positive doubled area."""
    n = len(points)
    cnt = Counter()
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if d > 0:
                    cnt[d] += 1
    return max(cnt.values()) if cnt else 0

def construct():
    t0 = time.time()
    DEADLINE = 75.0

    # Try several structured configurations
    candidates = []

    # 6x5 grid (seed)
    candidates.append([(x, y) for x in range(6) for y in range(5)])

    # 10x3 grid
    candidates.append([(x, y) for x in range(10) for y in range(3)])

    # 15x2 grid
    candidates.append([(x, y) for x in range(15) for y in range(2)])

    # 30x1 - all on a line, all zero area, bad
    # 5x6 grid
    candidates.append([(x, y) for x in range(5) for y in range(6)])

    # 4x7.5 not integer, try 4x8=32 too many, 3x10=30
    candidates.append([(x, y) for x in range(3) for y in range(10)])

    # Try a diamond/rhombus pattern
    # Points on two parallel lines with offset
    # Line 1: y=0, x = 0,2,4,...,28 (15 points)
    # Line 2: y=1, x = 1,3,5,...,29 (15 points)
    candidates.append([(2*i, 0) for i in range(15)] + [(2*i+1, 1) for i in range(15)])

    # Try 12x2.5 -> not integer. 12x2=24, need 30. 
    # Try a 6x5 but shifted
    # Try points forming a hexagonal lattice subset
    # Let's try: 3 rows of 10, but with alternating offsets
    # Row 0: x=0..9, y=0
    # Row 1: x=1..10, y=2  
    # Row 2: x=0..9, y=4
    candidates.append([(x, 0) for x in range(10)] + [(x, 2) for x in range(1, 11)] + [(x, 4) for x in range(10)])

    # Try 5 rows of 6 with different spacing
    # Dense cluster: 6x5 in small area
    candidates.append([(x, y) for x in range(6) for y in range(5)])

    # Try: two interleaved 15-point lines
    # y=0: x=0,1,...,14
    # y=2: x=0,1,...,14
    candidates.append([(x, 0) for x in range(15)] + [(x, 2) for x in range(15)])

    # Try: three interleaved 10-point lines
    candidates.append([(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)])

    # Try: 10x3 with wider spacing
    candidates.append([(x, 0) for x in range(10)] + [(x, 3) for x in range(10)] + [(x, 6) for x in range(10)])

    # Evaluate all candidates
    best_points = candidates[0]
    best_score = 0
    for pts in candidates:
        if len(pts) != 30:
            continue
        # Check all distinct and in range
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= x <= 30 and -30 <= y <= 30) for x, y in pts):
            continue
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts

    # If we have time, try some random perturbations of the best
    if time.time() - t0 < DEADLINE - 10:
        import random
        random.seed(42)
        for _ in range(500):
            if time.time() - t0 > DEADLINE - 5:
                break
            # Perturb best_points slightly
            pts = list(best_points)
            idx = random.randint(0, 29)
            dx, dy = random.randint(-3, 3), random.randint(-3, 3)
            new_pt = (pts[idx][0] + dx, pts[idx][1] + dy)
            if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                continue
            if new_pt in pts:
                continue
            pts[idx] = new_pt
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts

    return best_points

# EVOLVE-BLOCK-END
