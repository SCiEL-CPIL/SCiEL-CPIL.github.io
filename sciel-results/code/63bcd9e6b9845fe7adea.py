# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) to canonical form."""
    if A == 0 and B == 0:
        return None
    # Find gcd of |A|, |B|, |C|
    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g

    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C

    return (A, B, C)


def _line_from_points(p1, p2):
    """Get line through two points, normalized."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _count_incidences(points, lines):
    """Count point-line incidences exactly."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def _all_lines_through_points(points, max_points_per_line=54):
    """Find all lines that pass through 2+ points, with their incidence counts."""
    n = len(points)
    line_to_points = {}

    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line is None:
                continue
            if line not in line_to_points:
                line_to_points[line] = set()
            line_to_points[line].add(i)
            line_to_points[line].add(j)

    # Filter to lines with 2+ points (all should have at least 2)
    # Return list of (line, count) sorted by count descending
    result = [(line, len(pts)) for line, pts in line_to_points.items() if len(pts) >= 2]
    result.sort(key=lambda x: -x[1])
    return result


def _pick_top_k_lines(points, k=27):
    """Pick top k lines by incidence count."""
    all_lines = _all_lines_through_points(points)
    # Take top k, but if fewer than k, pad with arbitrary lines
    selected = all_lines[:k]
    if len(selected) < k:
        # This shouldn't happen with 54 points, but just in case
        # Add horizontal/vertical lines as fallback
        existing = set(line for line, _ in selected)
        for line in [(0, 1, -1), (1, 0, -1), (0, 1, -2), (1, 0, -2)]:
            if len(selected) >= k:
                break
            if line not in existing:
                selected.append((line, 0))
                existing.add(line)
    return [line for line, _ in selected[:k]]


def _local_search(points, lines, max_iterations=500, timeout=None, start_time=None):
    """Local search to improve incidence count."""
    if start_time is None:
        start_time = time.time()
    if timeout is None:
        timeout = 600.0

    current_points = np.array(points, dtype=np.int64).copy()
    current_lines = np.array(lines, dtype=np.int64).copy()
    current_inc = _count_incidences(current_points, current_lines)

    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_inc = current_inc

    n = len(current_points)
    m = len(current_lines)

    # Coordinate descent: try moving each point slightly
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    iteration = 0
    while iteration < max_iterations:
        if time.time() - start_time > timeout - 30:
            break

        improved = False
        # Try moving each point
        for i in range(n):
            if time.time() - start_time > timeout - 30:
                break
            for dx, dy in directions:
                new_points = current_points.copy()
                new_points[i] = [current_points[i][0] + dx, current_points[i][1] + dy]

                # Check for duplicate points
                if len(np.unique(new_points, axis=0)) < n:
                    continue

                # Recompute lines from points
                new_lines = _pick_top_k_lines(new_points, k=m)
                new_lines = np.array(new_lines, dtype=np.int64)

                new_inc = _count_incidences(new_points, new_lines)
                if new_inc > current_inc:
                    current_points = new_points
                    current_lines = new_lines
                    current_inc = new_inc
                    improved = True
                    if current_inc > best_inc:
                        best_inc = current_inc
                        best_points = current_points.copy()
                        best_lines = current_lines.copy()
                    break

        # Try swapping two points
        if not improved:
            for i in range(n):
                if time.time() - start_time > timeout - 30:
                    break
                for j in range(i + 1, n):
                    if time.time() - start_time > timeout - 30:
                        break
                    # Swap positions
                    new_points = current_points.copy()
                    new_points[i] = current_points[j]
                    new_points[j] = current_points[i]

                    if len(np.unique(new_points, axis=0)) < n:
                        continue

                    new_lines = _pick_top_k_lines(new_points, k=m)
                    new_lines = np.array(new_lines, dtype=np.int64)

                    new_inc = _count_incidences(new_points, new_lines)
                    if new_inc > current_inc:
                        current_points = new_points
                        current_lines = new_lines
                        current_inc = new_inc
                        improved = True
                        if current_inc > best_inc:
                            best_inc = current_inc
                            best_points = current_points.copy()
                            best_lines = current_lines.copy()
                        break
                if improved:
                    break

        if not improved:
            break

        iteration += 1

    return best_points, best_lines, best_inc


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    """
    start_time = time.time()
    deadline = start_time + 570  # Leave 30s safety margin

    k = 3
    # Points: {1,...,k} x {1,...,2*k^2}
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    # Lines: y = a*x + b  ->  a*x - y + b = 0  ->  (A, B, C) = (a, -1, b)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)

    # Verify baseline
    base_inc = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_inc = base_inc

    # Local search
    if time.time() < deadline - 60:
        sp, sl, si = _local_search(points, lines, max_iterations=200, timeout=deadline - start_time, start_time=start_time)
        if si > best_inc:
            best_points, best_lines, best_inc = sp, sl, si

    # Try a few different initial configurations
    configs = [
        # Grid 3x18
        lambda: (np.stack([np.tile([1,2,3], 18), np.repeat(np.arange(1,19), 3)], axis=1).astype(np.int64),
                 np.array([(a, -1, b) for a in range(1,4) for b in range(1,10)], dtype=np.int64)),
        # Grid 6x9
        lambda: (np.stack([np.tile(np.arange(1,7), 9), np.repeat(np.arange(1,10), 6)], axis=1).astype(np.int64),
                 np.array([(a, -1, b) for a in range(1,7) for b in range(1,5)], dtype=np.int64)),
    ]

    for cfg_func in configs:
        if time.time() > deadline - 60:
            break
        try:
            pts, lns = cfg_func()
            if len(pts) != 54 or len(lns) != 27:
                continue
            inc = _count_incidences(pts, lns)
            if inc > best_inc:
                best_points, best_lines, best_inc = pts.copy(), lns.copy(), inc
            # Local search from this config
            if time.time() < deadline - 30:
                sp, sl, si = _local_search(pts, lns, max_iterations=100, timeout=deadline - start_time, start_time=start_time)
                if si > best_inc:
                    best_points, best_lines, best_inc = sp, sl, si
        except Exception:
            continue

    return best_points, best_lines


# EVOLVE-BLOCK-END
