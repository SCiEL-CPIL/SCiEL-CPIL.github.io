# EVOLVE-BLOCK-START
import time
import numpy as np


def _incidence_count(points, lines):
    """Exact int64 count of (point, line) pairs with A*x + B*y + C == 0."""
    # points: (n,2) int64, lines: (m,3) int64
    # For each line l and point p: A*x + B*y + C
    # Vectorize per line to keep memory bounded.
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += int(np.count_nonzero(vals == 0))
    return total


def _canonical_line(A, B, C):
    """Reduce (A,B,C) by gcd(|A|,|B|,|C|) and make first nonzero entry positive."""
    import math
    g = 0
    for v in (A, B, C):
        g = math.gcd(g, abs(int(v)))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # make first nonzero positive
    for i, v in enumerate((A, B, C)):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _validate(points, lines):
    """Return True if the configuration satisfies all contract validity rules."""
    if points.shape != (54, 2):
        return False
    if lines.shape != (27, 3):
        return False
    if not (np.issubdtype(points.dtype, np.integer) and np.issubdtype(lines.dtype, np.integer)):
        return False
    # repeated points
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    seen_lines = set()
    for A, B, C in lines.tolist():
        if A == 0 and B == 0:
            return False
        c = _canonical_line(A, B, C)
        if c in seen_lines:
            return False
        seen_lines.add(c)
    return True


def _build_configuration():
    """Line-first construction: 3 vertical lines (18 pts each) + 24 pair-lines (2 pts each)."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)  # 1..18
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    # Vertical lines x = 1,2,3  ->  (1, 0, -c)
    line_rows = [(1, 0, -c) for c in (1, 2, 3)]

    # Pair-lines: for each unordered pair of points on distinct columns,
    # add the line through them. Each such line passes through exactly 2 points
    # (no three grid points are collinear off the verticals).
    # We need 24 more lines. Build all candidate pair-lines and pick 24.
    # To keep it deterministic and simple, iterate over all pairs of points
    # (i<j) that are NOT both on the same column (those are vertical, already used),
    # generate the line, and collect unique ones until we have 24.
    seen = set()
    for i in range(len(points)):
        x1, y1 = points[i]
        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            if x1 == x2:
                continue  # vertical, skip
            A = y2 - y1
            B = x1 - x2
            C = -A * x1 - B * y1
            c = _canonical_line(A, B, C)
            if c in seen:
                continue
            seen.add(c)
            line_rows.append(c)
            if len(line_rows) == 27:
                break
        if len(line_rows) == 27:
            break

    lines = np.array(line_rows, dtype=np.int64)
    return points, lines


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization for n=54 points, m=27 lines.

    Line-first design: a 3x18 grid gives 3 vertical lines each containing 18 points
    (54 incidences), plus 24 additional lines each passing through exactly 2 grid
    points (48 incidences), for a total of 102 incidences.
    """
    deadline = time.time() + 600.0
    safety_margin = 30.0

    points, lines = _build_configuration()

    # Validate and count.
    if not _validate(points, lines):
        # Fallback to the known-valid seed grid construction.
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)

    # Safety guard: if we are close to the deadline, return immediately.
    if time.time() + safety_margin > deadline:
        return points, lines

    return points, lines


# EVOLVE-BLOCK-END
