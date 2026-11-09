"""Combined program: structured candidates + target-area local search."""
import time
import itertools
from collections import Counter
import numpy as np


def _doubled_areas(points):
    """Compute all positive doubled areas for all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        ax, ay = points[i]
        for j in range(i + 1, n):
            bx, by = points[j]
            dx1 = bx - ax
            dy1 = by - ay
            for k in range(j + 1, n):
                cx, cy = points[k]
                dx2 = cx - ax
                dy2 = cy - ay
                det = abs(dx1 * dy2 - dy1 * dx2)
                if det > 0:
                    areas.append(det)
    return areas


def _area_counter(points):
    """Return Counter of positive doubled areas."""
    return Counter(_doubled_areas(points))


def _max_multiplicity(points):
    """Return the maximum count of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]


def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True


def _score(points):
    if not _valid(points):
        return 0
    return _max_multiplicity(points)


def _fast_score_target(points, prev_count, moved_idx, old_point, new_point, target_area):
    """
    Fast scoring for a specific target area.
    Returns the new count of triples with area == target_area.
    """
    n = len(points)
    idx = moved_idx
    count = prev_count

    # Remove old contributions (triples containing idx with old_point)
    ox, oy = old_point
    for i in range(n):
        if i == idx:
            continue
        px, py = points[i]
        for j in range(i + 1, n):
            if j == idx:
                continue
            qx, qy = points[j]
            det = abs((px - ox) * (qy - oy) - (py - oy) * (qx - ox))
            if det == target_area:
                count -= 1

    # Add new contributions (triples containing idx with new_point)
    nx, ny = new_point
    for i in range(n):
        if i == idx:
            continue
        px, py = points[i]
        for j in range(i + 1, n):
            if j == idx:
                continue
            qx, qy = points[j]
            det = abs((px - nx) * (qy - ny) - (py - ny) * (qx - nx))
            if det == target_area:
                count += 1

    return count


def _structured_candidates():
    """Generate various structured grid-like candidates."""
    candidates = []

    # Standard grids: m x n with m*n = 30
    for m, n in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]:
        pts = []
        for i in range(m):
            for j in range(n):
                x = i - m // 2
                y = j - n // 2
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)

    # Grids with different spacings
    for m, n in [(6, 5), (5, 6), (10, 3)]:
        for sx in [1, 2, 3]:
            for sy in [1, 2, 3]:
                pts = []
                for i in range(m):
                    for j in range(n):
                        x = i * sx - (m * sx) // 2
                        y = j * sy - (n * sy) // 2
                        pts.append((x, y))
                if _valid(pts):
                    candidates.append(pts)

    # Interleaved: points on lines
    for offset in range(0, 5):
        pts = []
        for line in range(3):
            for k in range(10):
                x = k * 6 - 30 + offset
                y = line * 10 - 10
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)

    # Staggered rows (hexagonal-ish)
    for dx in [1, 2, 3]:
        for dy in [1, 2, 3]:
            pts = []
            for row in range(5):
                for col in range(6):
                    x = col * dx + (row % 2) * (dx // 2 + 1) - 15
                    y = row * dy - 10
                    pts.append((x, y))
            if _valid(pts):
                candidates.append(pts)

    # Arithmetic progression in both directions
    for step_x in [1, 2, 3]:
        for step_y in [1, 2, 3]:
            for start_x in range(-20, 20, 2):
                for start_y in range(-20, 20, 2):
                    pts = []
                    for i in range(30):
                        row = i // 6
                        col = i % 6
                        x = start_x + col * step_x
                        y = start_y + row * step_y
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            pts.append((x, y))
                    if len(pts) == 30 and _valid(pts):
                        candidates.append(pts)

    # Random perturbations of the best grid
    best_grid = [(x, y) for x in range(-2, 3) for y in range(-2, 3)] # 5x5 not 30, use 6x5
    best_grid = [(x, y) for x in range(-2, 4) for y in range(-2, 3)] # 6x5=30
    for _ in range(20):
        pts = list(best_grid)
        np.random.shuffle(pts)
        # Perturb
        for i in range(30):
            if np.random.rand() < 0.2:
                dx = np.random.randint(-2, 3)
                dy = np.random.randint(-2, 3)
                nx, ny = pts[i][0] + dx, pts[i][1] + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    if (nx, ny) not in pts:
                        pts[i] = (nx, ny)
        if _valid(pts):
            candidates.append(pts)

    return candidates


def _local_search_target(points, target_area, time_limit=10.0):
    """
    Local search to maximize the number of triples with a specific target_area.
    """
    start_time = time.time()
    best_points = list(points)
    occupied = set(best_points)
    
    # Initial count
    best_count = 0
    for i in range(30):
        ax, ay = best_points[i]
        for j in range(i + 1, 30):
            bx, by = best_points[j]
            dx1 = bx - ax
            dy1 = by - ay
            for k in range(j + 1, 30):
                cx, cy = best_points[k]
                dx2 = cx - ax
                dy2 = cy - ay
                det = abs(dx1 * dy2 - dy1 * dx2)
                if det == target_area:
                    best_count += 1

    neighborhoods = [
        [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)],
        [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if (dx, dy) != (0, 0)],
    ]

    improved = True
    while improved and time.time() - start_time < time_limit - 1:
        improved = False
        for idx in range(30):
            if time.time() - start_time > time_limit - 1:
                break
            px, py = best_points[idx]
            for nh in neighborhoods:
                for dx, dy in nh:
                    nx, ny = px + dx, py + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    new_point = (nx, ny)
                    if new_point in occupied:
                        continue
                    
                    new_points = list(best_points)
                    new_points[idx] = new_point
                    new_count = _fast_score_target(new_points, best_count, idx, (px, py), new_point, target_area)
                    
                    if new_count > best_count:
                        best_count = new_count
                        best_points = new_points
                        occupied.discard((px, py))
                        occupied.add(new_point)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_points, best_count


def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s

    # Generate structured candidates
    candidates = _structured_candidates()

    if not candidates:
        # Fallback: 6x5 grid
        return [(x, y) for x in range(6) for y in range(5)]

    # Score all candidates to find a good starting point
    best_points = None
    best_score = 0
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)

    if best_points is None:
        best_points = list(candidates[0])
        best_score = _score(best_points)

    # Determine candidate target areas
    # We look at the areas present in the best candidate and some small integers
    counter = _area_counter(best_points)
    candidate_areas = set(counter.keys())
    # Add small integers
    for a in range(1, 20):
        candidate_areas.add(a)
    
    # Sort by frequency in best candidate (most frequent first)
    sorted_areas = sorted(candidate_areas, key=lambda a: counter.get(a, 0), reverse=True)
    
    # Limit the number of target areas to search to fit time budget
    # We have ~50s left. Each target search takes ~5-10s. So we can try ~5-10 targets.
    num_targets = 8
    if len(sorted_areas) > num_targets:
        sorted_areas = sorted_areas[:num_targets]
    
    # Perform local search for each target area
    for area in sorted_areas:
        if time.time() > deadline - 15:
            break
        remaining_time = deadline - time.time() - 5
        if remaining_time < 5:
            break
        
        result_points, result_count = _local_search_target(best_points, area, time_limit=remaining_time)
        
        # Check if this result is better than the global best
        # The score is the max multiplicity, so if result_count > best_score, it's an improvement
        # But we must ensure result_count is indeed the max multiplicity for result_points
        # It's possible that another area has a higher count in result_points.
        # So we must recalculate the full score for the new points.
        full_score = _score(result_points)
        if full_score > best_score:
            best_score = full_score
            best_points = result_points

    return best_points
