"""
Minimize ordinary lines (lines containing exactly 2 of the 12 points)
for 12 integer points with |x|,|y| <= 1e9.

Strategy:
1. Use known combinatorial/geometric constructions that minimize ordinary lines.
2. For n=12, the theoretical minimum over reals is 6 (from the Orchard-Planting problem / Green-Tao related results).
3. We try to find a configuration of 12 rational/integer points achieving as few ordinary lines as possible.

Known approach: Use points on a conic or structured arrangements.
One classic construction uses points on a parabola or a grid with careful selection.

Actually, for small n, the minimum number of ordinary lines is known:
n=3: 3
n=4: 3 (if not all collinear; a parallelogram has 3 ordinary lines? Let's check: 4 points in convex position - each pair defines a line, 6 lines total, but in a parallelogram, 2 pairs are parallel but not collinear... wait, each line through 2 points: in a general quadrilateral, all 6 lines are ordinary. In a parallelogram, the diagonals intersect but don't create 3-point lines. So still 6? No wait, 4 points: C(4,2)=6 pairs. If no 3 are collinear, all 6 lines are ordinary. But we can have 3 collinear: then the line through those 3 is not ordinary, and the other lines... 

Let me think about known minima. The problem of minimum ordinary lines for n points:
- n=3: 3
- n=4: 3 (take 3 collinear + 1 off: the line with 3 is not ordinary; the 3 lines from the 4th point to each of the 3 are ordinary if the 4th isn't on those lines... wait, if 3 are collinear and 1 is off, then lines from the off-point to each of the 3: these 3 lines each contain exactly 2 points (the off-point and one of the 3), so 3 ordinary lines. The line through the 3 collinear points is not ordinary. So 3 ordinary lines.)
- n=5: 4? 
- Actually the known sequence for minimum ordinary lines: 3,3,4,6,6,8,... for n=3,4,5,6,7,8...

For n=12, I believe the minimum might be around 6-10.

Let me use a different approach: 
- Start with a good initial configuration
- Use local search (move points to reduce ordinary line count)
- Use simulated annealing or hill climbing

But the key insight: we need integer points. Let's use a structured approach.

One idea: use points on a small grid and then optimize.
Another: use the fact that if many points are collinear, we reduce ordinary lines.

Let me try a construction: 
- Put points on lines that intersect at common points.

Actually, let me use a computational approach:
1. Generate candidate configurations from known patterns
2. Use local search to improve

For the local search, I'll represent points as integers and try small perturbations.

Let me code a solution that:
1. Tries several initial configurations
2. Uses local search to minimize ordinary lines
3. Returns the best found

Given the 150s deadline, I'll use time-bounded search.
"""

import time
import random
import math
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the points.

    Uses the pair-count inversion: a line through k points contributes
    C(k,2) pairs.  A line is ordinary iff exactly 2 points lie on it,
    which means it contributes exactly C(2,2)=1 pair.  So the number of
    ordinary lines equals the number of distinct lines that appear
    exactly once in the pair enumeration.
    """
    n = len(points)
    if n < 3:
        return 0

    line_pair_count = {}

    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1

            A = dy
            B = -dx
            C = dx * y1 - dy * x1

            g = math.gcd(abs(A), abs(B))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g

            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C

            key = (A, B, C)
            line_pair_count[key] = line_pair_count.get(key, 0) + 1

    return sum(1 for c in line_pair_count.values() if c == 1)


def _are_all_collinear(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        x2, y2 = points[i]
        if (x2 - x0) * (y1 - y0) != (x1 - x0) * (y2 - y0):
            return False
    return True


def _local_search(points, max_iters=5000, deadline=None):
    """Local search to minimize ordinary lines.

    Uses two move types:
      1. Single-point relocation (small random perturbation)
      2. Pair-swap: exchange the positions of two points (or move one
         point to the other's position and vice-versa with a perturbation).
         Cell findings indicate pair-swap explores a richer space.
    """
    n = len(points)
    best = [list(p) for p in points]
    best_score = _count_ordinary_lines(best)

    if deadline is not None and time.time() > deadline:
        return best, best_score

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        move_type = random.random()

        if move_type < 0.5:
            # --- Single-point move ---
            idx = random.randint(0, n - 1)
            old_x, old_y = best[idx]
            step = random.choice([1, 1, 2, 3, 5, 10, 20, 50])
            new_x = old_x + random.randint(-step, step)
            new_y = old_y + random.randint(-step, step)

            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue

            new_points = [list(p) for p in best]
            new_points[idx] = [new_x, new_y]

            # Check distinctness
            seen = set()
            valid = True
            for p in new_points:
                t = (p[0], p[1])
                if t in seen:
                    valid = False
                    break
                seen.add(t)
            if not valid:
                continue

            if _are_all_collinear(new_points):
                continue

            score = _count_ordinary_lines(new_points)
            if score < best_score:
                best = new_points
                best_score = score

        else:
            # --- Pair-swap / pair-relocate move ---
            i, j = random.sample(range(n), 2)
            xi, yi = best[i]
            xj, yj = best[j]

            mode = random.random()
            if mode < 0.3:
                # Pure swap
                new_points = [list(p) for p in best]
                new_points[i] = [xj, yj]
                new_points[j] = [xi, yi]
            elif mode < 0.6:
                # Move point i to a perturbation of point j's position
                step = random.choice([1, 2, 3, 5])
                new_points = [list(p) for p in best]
                new_points[i] = [xj + random.randint(-step, step),
                                 yj + random.randint(-step, step)]
            else:
                # Move both points to new random positions near each other
                step = random.choice([1, 2, 3, 5, 10])
                new_points = [list(p) for p in best]
                new_points[i] = [xi + random.randint(-step, step),
                                 yi + random.randint(-step, step)]
                new_points[j] = [xj + random.randint(-step, step),
                                 yj + random.randint(-step, step)]

            # Bounds check
            if abs(new_points[i][0]) > 10**9 or abs(new_points[i][1]) > 10**9:
                continue
            if abs(new_points[j][0]) > 10**9 or abs(new_points[j][1]) > 10**9:
                continue

            # Distinctness
            seen = set()
            valid = True
            for p in new_points:
                t = (p[0], p[1])
                if t in seen:
                    valid = False
                    break
                seen.add(t)
            if not valid:
                continue

            if _are_all_collinear(new_points):
                continue

            score = _count_ordinary_lines(new_points)
            if score < best_score:
                best = new_points
                best_score = score

    return best, best_score


def _generate_structured_candidates():
    """Generate a diverse set of structured and semi-structured candidates."""
    candidates = []

    # 1. 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])

    # 2. 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])

    # 3. Near-pencil: 11 on a line + 1 off
    pts = [[i, 0] for i in range(11)] + [[0, 1]]
    candidates.append(pts)

    # 4. 6+6: two lines of 6, offset
    pts = [[i, 0] for i in range(6)] + [[i, 3] for i in range(6)]
    candidates.append(pts)

    # 5. 4+4+4: three parallel lines of 4
    pts = [[i, 0] for i in range(4)] + [[i, 2] for i in range(4)] + [[i, 4] for i in range(4)]
    candidates.append(pts)

    # 6. 5+4+3: three parallel lines
    pts = [[i, 0] for i in range(5)] + [[i, 2] for i in range(4)] + [[i, 4] for i in range(3)]
    candidates.append(pts)

    # 7. Two intersecting lines: 6 on x-axis, 6 on y-axis, sharing origin
    #    But we need 12 distinct points, so 5+1+5+1 = 12 (origin counted once)
    pts = [[i, 0] for i in range(1, 6)] + [[0, i] for i in range(1, 6)] + [[0, 0], [5, 5]]
    # That's 10 + 2 = 12
    candidates.append(pts)

    # 8. Diagonal-shifted grid: (i, j+i) for i in 0..3, j in 0..2
    pts = [[i, j + i] for i in range(4) for j in range(3)]
    candidates.append(pts)

    # 9. Parabolic: (i, i*i) for i in 0..11
    pts = [[i, i * i] for i in range(12)]
    candidates.append(pts)

    # 10. Hexagonal-ish: points on a small hexagonal lattice
    pts = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if abs(dx) + abs(dy) <= 3 and (dx, dy) not in [(0, 0)]:
                pts.append([dx * 2, dy * 2 + dx])
            if len(pts) >= 12:
                break
        if len(pts) >= 12:
            break
    if len(pts) == 12 and not _are_all_collinear(pts):
        candidates.append(pts)

    # 11. 3 collinear triples on 3 different lines through a common point
    #     Lines: horizontal, vertical, diagonal (slope 1)
    #     Points on horizontal: (-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0)  -> 6 pts
    #     Points on vertical: (0,-3),(0,-2),(0,-1),(0,1),(0,2),(0,3)  -> 6 pts (origin shared)
    #     That's 11 distinct. Add one more on diagonal: (3,3)
    pts = [[i, 0] for i in range(-3, 3)] + [[0, i] for i in range(-3, 3) if i != 0] + [[3, 3]]
    if len(pts) == 12:
        candidates.append(pts)

    # 12. 4 lines of 3 through a common point (like a star)
    #     Directions: (1,0), (0,1), (1,1), (1,-1)
    pts = []
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        for t in [-1, 1, 2]:
            p = [t * dx, t * dy]
            if p not in pts:
                pts.append(p)
    # That gives 12 points (no overlap since directions are distinct and t values are ±1,2)
    if len(pts) == 12 and not _are_all_collinear(pts):
        candidates.append(pts)

    # 13. Random small-grid points (several seeds)
    random.seed(42)
    for trial in range(30):
        pts = []
        for _ in range(12):
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            if [x, y] not in pts:
                pts.append([x, y])
        while len(pts) < 12:
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 12 and not _are_all_collinear(pts):
            candidates.append(pts)

    # 14. Random with collinear bias: pick a line, put several points on it
    random.seed(123)
    for trial in range(20):
        # Pick a random line direction
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            dx, dy = 1, 0
        # Pick a base point
        bx, by = random.randint(-3, 3), random.randint(-3, 3)
        # Put 5-7 points on this line
        k_on_line = random.randint(5, 7)
        pts = []
        for t in range(k_on_line):
            p = [bx + t * dx, by + t * dy]
            if p not in pts:
                pts.append(p)
        # Fill remaining randomly
        while len(pts) < 12:
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 12 and not _are_all_collinear(pts):
            candidates.append(pts)

    return candidates


def _try_construction():
    """Try various constructions and return the best configuration."""
    candidates = _generate_structured_candidates()

    best_points = None
    best_score = float('inf')

    deadline = time.time() + 130  # Leave margin for the 150s limit

    # Sort candidates by initial score (cheapest first) so we spend more
    # time polishing the most promising ones
    scored = []
    for cand in candidates:
        s = _count_ordinary_lines(cand)
        scored.append((s, cand))
    scored.sort(key=lambda x: x[0])

    # Allocate iterations: give more to better initial candidates
    # Use a simple scheme: top 3 get 4000 iters, next 5 get 2000, rest get 800
    n_cands = len(scored)
    for rank, (init_score, cand) in enumerate(scored):
        if time.time() > deadline:
            break

        if rank < 3:
            max_iters = 4000
        elif rank < 8:
            max_iters = 2000
        else:
            max_iters = 800

        optimized, score = _local_search(cand, max_iters=max_iters, deadline=deadline)

        if score < best_score:
            best_score = score
            best_points = optimized

        # If we've hit the theoretical lower bound, stop early
        if best_score <= 6:
            break

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)

    return best_points, best_score


def construct():
    """Construct 12 integer points minimizing ordinary lines."""
    start = time.time()
    
    points, score = _try_construction()
    
    # Ensure validity
    assert len(points) == 12
    assert len(set(tuple(p) for p in points)) == 12
    for p in points:
        assert abs(p[0]) <= 10**9
        assert abs(p[1]) <= 10**9
    assert not _are_all_collinear(points)
    
    return points
