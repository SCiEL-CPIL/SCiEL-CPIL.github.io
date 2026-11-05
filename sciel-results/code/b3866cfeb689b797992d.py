"""Optimized search for 16 integer points maximizing lines with exactly 4 points."""
import time
import random
from math import gcd

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _line_key(x1, y1, x2, y2):
    """Canonical key for the line through two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', x1)
    if dy == 0:
        return ('h', y1)
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # line: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    return ('s', dx, dy, c)


def _count_four_lines(points):
    """Count lines containing exactly 4 of the points. Also check no 5 collinear."""
    line_counts = {}
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            key = _line_key(xi, yi, xj, yj)
            line_counts[key] = line_counts.get(key, 0) + 1
    # A line with k points has C(k,2) pairs
    # We need to find lines with exactly 4 points
    # Count pairs per line, then invert to find k
    four_count = 0
    five_violation = False
    for key, pair_count in line_counts.items():
        # Find k such that C(k,2) = pair_count
        # C(k,2) = k*(k-1)/2
        # For small k: k=2->1, k=3->3, k=4->6, k=5->10, k=6->15, k=7->21
        if pair_count == 6:
            four_count += 1
        elif pair_count >= 10:
            five_violation = True
    if five_violation:
        return -1  # Invalid
    return four_count


def _evaluate(points):
    """Return number of 4-point lines, or -1 if invalid."""
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 16:
        return -1
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
    return _count_four_lines(points)


def _affine_transform(points, a, b, c, d, e, f):
    """Apply affine transform to points."""
    result = []
    for x, y in points:
        nx = a * x + b * y + c
        ny = c * x + d * y + e  # Wait, let me fix this
        result.append([nx, ny])
    return result


def _random_affine(points, scale=50):
    """Apply a random affine transformation."""
    a = random.randint(-scale, scale)
    b = random.randint(-scale, scale)
    c = random.randint(-scale, scale)
    d = random.randint(-scale, scale)
    e = random.randint(-scale, scale)
    f = random.randint(-scale, scale)
    # Ensure it's invertible (not degenerate)
    if a * d - b * c == 0:
        return None
    result = []
    for x, y in points:
        nx = a * x + b * y + c
        ny = d * x + e * y + f
        result.append([nx, ny])
    return result


def construct():
    best_score = 0
    best_points = None
    deadline = _START_TIME + _DEADLINE

    # Initial: 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    score = _evaluate(grid)
    if score > best_score:
        best_score = score
        best_points = [list(p) for p in grid]

    # Try various transformations and perturbations
    random.seed(42)

    for iteration in range(1000000):
        if time.time() > deadline:
            break

        # Start from a random base
        if random.random() < 0.5:
            # Start from grid
            base = [list(p) for p in grid]
        else:
            # Start from best found
            base = [list(p) for p in best_points] if best_points else [list(p) for p in grid]

        # Apply random affine transform
        transformed = _random_affine(base, scale=30)
        if transformed is None:
            continue

        # Check bounds
        if any(abs(x) > 1000 or abs(y) > 1000 for x, y in transformed):
            continue

        score = _evaluate(transformed)
        if score > best_score:
            best_score = score
            best_points = [list(p) for p in transformed]

    # Local search: try moving individual points
    if best_points is not None:
        current = [list(p) for p in best_points]
        current_score = _evaluate(current)

        for step in range(500000):
            if time.time() > deadline:
                break

            # Pick a random point to move
            idx = random.randint(0, 15)
            old_x, old_y = current[idx]

            # Try a random new position near the old one
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_x = old_x + dx
            new_y = old_y + dy

            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue

            current[idx] = [new_x, new_y]
            new_score = _evaluate(current)

            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current]
            else:
                # Revert
                current[idx] = [old_x, old_y]

    if best_points is None:
        best_points = grid

    return best_points
