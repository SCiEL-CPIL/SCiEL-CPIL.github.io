import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    def score(points):
        """Compute maximum multiplicity of any single positive doubled area."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                a = pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
                b = pts[j, 0] - pts[i, 0]
                c = pts[i, 1] - pts[j, 1]
                if j + 1 < n:
                    yk = pts[j+1:, 1]
                    xk = pts[j+1:, 0]
                    vals = np.abs(a + yk * b + xk * c)
                    areas.append(vals)
        if not areas:
            return 0
        all_areas = np.concatenate(areas)
        all_areas = all_areas[all_areas > 0]
        if len(all_areas) == 0:
            return 0
        unique, counts = np.unique(all_areas, return_counts=True)
        return int(counts.max())

    def is_valid(points):
        """Check all points are distinct and in [-30,30]^2."""
        if len(points) != 30:
            return False
        s = set(points)
        if len(s) != 30:
            return False
        for x, y in points:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return False
        return True

    # Generate structured candidates
    candidates = []

    # 6x5 grid (seed)
    grid_6x5 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(grid_6x5)

    # 5x5 grid + 5 extra points
    grid_5x5 = [(x, y) for x in range(5) for y in range(5)]
    candidates.append(grid_5x5 + [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)])
    candidates.append(grid_5x5 + [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4)])
    candidates.append(grid_5x5 + [(5, 0), (5, 2), (5, 4), (7, 0), (7, 4)])
    candidates.append(grid_5x5 + [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)])
    candidates.append(grid_5x5 + [(-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4)])

    # 4x4 grid + 14 extra
    grid_4x4 = [(x, y) for x in range(4) for y in range(4)]
    candidates.append(grid_4x4 + [(4, y) for y in range(5)] + [(5, y) for y in range(4)] + [(6, y) for y in range(3)] + [(0, 4), (0, 5)])

    # 10x3 grid
    grid_10x3 = [(x, y) for x in range(10) for y in range(3)]
    candidates.append(grid_10x3)

    # 15x2 grid
    grid_15x2 = [(x, y) for x in range(15) for y in range(2)]
    candidates.append(grid_15x2)

    # 30x1 line (all collinear, score 0 - skip but include for completeness)
    # grid_30x1 = [(x, 0) for x in range(30)]
    # candidates.append(grid_30x1)

    # 8x4 grid
    grid_8x4 = [(x, y) for x in range(8) for y in range(4)]
    candidates.append(grid_8x4)

    # 7x5 grid (35 points, take 30)
    grid_7x5 = [(x, y) for x in range(7) for y in range(5)]
    candidates.append(grid_7x5[:30])

    # Centered grids
    for cx, cy in [(-1, -1), (0, 0)]:
        g = [(cx + x, cy + y) for x in range(6) for y in range(5)]
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in g):
            candidates.append(g)

    # Evaluate all candidates
    best_score = 0
    best_points = candidates[0]

    for cand in candidates:
        if not is_valid(cand):
            continue
        s = score(cand)
        if s > best_score:
            best_score = s
            best_points = cand[:]

    # Local search: try moving individual points
    all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    random.shuffle(all_positions)

    # Build a set for fast lookup
    point_set = set(best_points)

    iterations = 0
    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline:
        iterations += 1

        if no_improve_count > max_no_improve:
            # Try a different starting structure
            no_improve_count = 0
            # Try a random structured candidate
            if candidates:
                start_idx = random.randint(0, len(candidates) - 1)
                best_points = candidates[start_idx][:]
                best_score = score(best_points)
                point_set = set(best_points)
            continue

        # Pick a random point to move
        idx = random.randint(0, 29)
        old_point = best_points[idx]
        point_set.discard(old_point)

        found_improvement = False
        # Try several candidate positions
        for attempt in range(20):
            # Generate a candidate near the old point or random
            if random.random() < 0.5:
                # Near old point
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                new_x = old_point[0] + dx
                new_y = old_point[1] + dy
            else:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)

            new_point = (new_x, new_y)

            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                continue
            if new_point in point_set:
                continue

            point_set.add(new_point)
            best_points[idx] = new_point

            s = score(best_points)

            if s > best_score:
                best_score = s
                found_improvement = True
                break
            else:
                # Revert
                best_points[idx] = old_point
                point_set.discard(new_point)

        if not found_improvement:
            # Revert if we didn't break with improvement
            best_points[idx] = old_point
            point_set.discard(best_points[idx])
            point_set.add(old_point)
            no_improve_count += 1
        else:
            no_improve_count = 0

    # Final validation
    if not is_valid(best_points):
        return candidates[0][:]

    return best_points
