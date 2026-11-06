"""
Minimize ordinary lines (lines containing exactly 2 of the 12 points)
for 12 integer points with |x|,|y| <= 1e9.

Strategy:
1. Use known combinatorial/geometric constructions that minimize ordinary lines.
2. For n=12, the theoretical minimum over reals is 6.
3. We try to find a configuration of 12 rational/integer points achieving as few ordinary lines as possible.

Approach:
- Generate structured candidates (grids, pencils, conics, etc.)
- Use a fast exact counter for ordinary lines.
- Run a multi-strategy local search with:
  * Single-point moves (small perturbations)
  * Collinear placement moves (move a point onto a line through two others)
  * Pair-swap moves
- Deadline-guarded to stay within 150s.
"""

import time
import random
import math
from itertools import combinations
from math import gcd
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the points.
    
    Uses line key normalization: for each pair, compute the line (A,B,C)
    in normalized form, track how many points lie on each line.
    A line is ordinary iff exactly 2 points lie on it.
    """
    n = len(points)
    if n < 3:
        return 0

    line_points = defaultdict(set)

    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1

            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g

            # Normalize direction
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy

            A = dy
            B = -dx
            C = dx * y1 - dy * x1

            # Normalize (A, B, C)
            g2 = gcd(abs(A), abs(B))
            if g2 == 0:
                continue
            A //= g2
            B //= g2
            C //= g2

            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C

            key = (A, B, C)
            line_points[key].add(i)
            line_points[key].add(j)

    return sum(1 for pts in line_points.values() if len(pts) == 2)


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


def _local_search(points, max_iters=5000, deadline=None, rng=None):
    """Local search to minimize ordinary lines.
    
    Uses three move types:
      1. Single-point relocation (small random perturbation)
      2. Collinear placement: move a point onto a line through two other points
      3. Pair-swap / pair-relocate move
    """
    if rng is None:
        rng = random.Random()
    
    n = len(points)
    best = [list(p) for p in points]
    best_score = _count_ordinary_lines(best)

    if deadline is not None and time.time() > deadline:
        return best, best_score

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        move_type = rng.random()

        if move_type < 0.4:
            # --- Single-point move (small perturbation) ---
            idx = rng.randint(0, n - 1)
            old_x, old_y = best[idx]
            step = rng.choice([1, 1, 2, 3, 5, 10, 20, 50])
            new_x = old_x + rng.randint(-step, step)
            new_y = old_y + rng.randint(-step, step)

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

        elif move_type < 0.75:
            # --- Collinear placement: move point idx onto line through points i, j ---
            idx = rng.randint(0, n - 1)
            others = [k for k in range(n) if k != idx]
            i, j = rng.sample(others, 2)
            p1 = best[i]
            p2 = best[j]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue
                
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Try positions on this line: p1 + k * (dx_n, dy_n)
            found_improvement = False
            for k in list(range(-8, 9)) + [rng.randint(-20, 20) for _ in range(4)]:
                if k == 0:
                    continue
                nx = p1[0] + k * dx_n
                ny = p1[1] + k * dy_n
                
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                
                new_p = [nx, ny]
                
                # Check if new_p is already in best (other than idx)
                occupied = False
                for m in range(n):
                    if m != idx and best[m] == new_p:
                        occupied = True
                        break
                if occupied:
                    continue
                
                new_points = [list(p) for p in best]
                new_points[idx] = new_p
                
                if _are_all_collinear(new_points):
                    continue
                    
                new_score = _count_ordinary_lines(new_points)
                
                if new_score < best_score:
                    best = new_points
                    best_score = new_score
                    found_improvement = True
                    if best_score <= 6:
                        return best, best_score
                    break  # Move to next iteration if improved
                
        else:
            # --- Pair-swap / pair-relocate move ---
            i, j = rng.sample(range(n), 2)
            xi, yi = best[i]
            xj, yj = best[j]

            mode = rng.random()
            if mode < 0.3:
                # Pure swap
                new_points = [list(p) for p in best]
                new_points[i] = [xj, yj]
                new_points[j] = [xi, yi]
            elif mode < 0.6:
                # Move point i to a perturbation of point j's position
                step = rng.choice([1, 2, 3, 5])
                new_points = [list(p) for p in best]
                new_points[i] = [xj + rng.randint(-step, step),
                                 yj + rng.randint(-step, step)]
            else:
                # Move both points to new random positions near each other
                step = rng.choice([1, 2, 3, 5, 10])
                new_points = [list(p) for p in best]
                new_points[i] = [xi + rng.randint(-step, step),
                                 yi + rng.randint(-step, step)]
                new_points[j] = [xj + rng.randint(-step, step),
                                 yj + rng.randint(-step, step)]

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

    # 7. Two intersecting lines: 5 on x-axis, 5 on y-axis, + origin + (5,5)
    pts = [[i, 0] for i in range(1, 6)] + [[0, i] for i in range(1, 6)] + [[0, 0], [5, 5]]
    candidates.append(pts)

    # 8. Diagonal-shifted grid: (i, j+i) for i in 0..3, j in 0..2
    pts = [[i, j + i] for i in range(4) for j in range(3)]
    candidates.append(pts)

    # 9. Parabolic: (i, i*i) for i in 0..11
    pts = [[i, i * i] for i in range(12)]
    candidates.append(pts)

    # 10. 3 collinear triples on 3 different lines through a common point
    pts = [[i, 0] for i in range(-3, 3)] + [[0, i] for i in range(-3, 3) if i != 0] + [[3, 3]]
    if len(pts) == 12:
        candidates.append(pts)

    # 11. 4 lines of 3 through a common point (like a star)
    pts = []
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        for t in [-1, 1, 2]:
            p = [t * dx, t * dy]
            if p not in pts:
                pts.append(p)
    if len(pts) == 12 and not _are_all_collinear(pts):
        candidates.append(pts)

    # 12. 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])

    # 13. 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])

    # 14. 5x3 grid (first 12)
    candidates.append([[x, y] for x in range(5) for y in range(3)][:12])

    # 15. 4x4 minus corners
    pts = [[x, y] for x in range(4) for y in range(4)]
    for corner in [[0, 0], [3, 0], [0, 3], [3, 3]]:
        if corner in pts:
            pts.remove(corner)
    candidates.append(pts)

    # 16. Hexagonal-like: two rows with offset
    pts_hex = []
    for x in range(6):
        pts_hex.append([2 * x, 0])
        pts_hex.append([2 * x + 1, 1])
    if len(pts_hex) == 12:
        candidates.append(pts_hex)

    # 17. Random small-grid points (several seeds)
    rng = random.Random(42)
    for trial in range(30):
        pts = []
        for _ in range(12):
            x = rng.randint(-4, 4)
            y = rng.randint(-4, 4)
            if [x, y] not in pts:
                pts.append([x, y])
        while len(pts) < 12:
            x = rng.randint(-8, 8)
            y = rng.randint(-8, 8)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 12 and not _are_all_collinear(pts):
            candidates.append(pts)

    # 18. Random with collinear bias
    rng = random.Random(123)
    for trial in range(20):
        dx = rng.randint(-3, 3)
        dy = rng.randint(-3, 3)
        if dx == 0 and dy == 0:
            dx, dy = 1, 0
        bx, by = rng.randint(-3, 3), rng.randint(-3, 3)
        k_on_line = rng.randint(5, 7)
        pts = []
        for t in range(k_on_line):
            p = [bx + t * dx, by + t * dy]
            if p not in pts:
                pts.append(p)
        while len(pts) < 12:
            x = rng.randint(-8, 8)
            y = rng.randint(-8, 8)
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

    deadline = time.time() + 135  # Leave margin for the 150s limit

    # Sort candidates by initial score (cheapest first)
    scored = []
    for cand in candidates:
        s = _count_ordinary_lines(cand)
        scored.append((s, cand))
    scored.sort(key=lambda x: x[0])

    # Allocate iterations: give more to better initial candidates
    n_cands = len(scored)
    rng = random.Random(2024)
    
    for rank, (init_score, cand) in enumerate(scored):
        if time.time() > deadline:
            break

        if rank < 3:
            max_iters = 5000
        elif rank < 8:
            max_iters = 2500
        else:
            max_iters = 1000

        optimized, score = _local_search(cand, max_iters=max_iters, deadline=deadline, rng=rng)

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
