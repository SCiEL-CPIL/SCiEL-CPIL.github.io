# EVOLVE-BLOCK-START
import numpy as np
from math import gcd


def _gcd3(a, b, c):
    return gcd(gcd(abs(a), abs(b)), abs(c))


def _normalize_line(a, b, c):
    g = _gcd3(a, b, c)
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _count_incidents(points, lines):
    total = 0
    for (x, y) in points:
        for (A, B, C) in lines:
            if A * x + B * y + C == 0:
                total += 1
    return total


def _validate(points, lines):
    if len(points) != 54 or len(lines) != 27:
        return False
    pt_set = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = int(p[0]), int(p[1])
        if (x, y) in pt_set:
            return False
        pt_set.add((x, y))
    ln_set = set()
    for l in lines:
        if len(l) != 3:
            return False
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        key = _normalize_line(A, B, C)
        if key is None:
            return False
        if key in ln_set:
            return False
        ln_set.add(key)
    return True


def _build_candidate(grid, line_specs):
    xs = grid[0]
    ys = grid[1]
    pts = []
    for x in xs:
        for y in ys:
            pts.append((x, y))
    lines = []
    for spec in line_specs:
        if spec[0] == 'v':
            c = spec[1]
            lines.append((1, 0, -c))
        elif spec[0] == 'h':
            c = spec[1]
            lines.append((0, 1, -c))
        elif spec[0] == 's1':
            c = spec[1]
            lines.append((1, -1, c))
        elif spec[0] == 's-1':
            c = spec[1]
            lines.append((1, 1, -c))
        elif spec[0] == 's2':
            c = spec[1]
            lines.append((2, -1, -c))
        elif spec[0] == 's-2':
            c = spec[1]
            lines.append((2, 1, -c))
        elif spec[0] == 's1/2':
            c = spec[1]
            lines.append((1, -2, c))
        elif spec[0] == 's-1/2':
            c = spec[1]
            lines.append((1, 2, -c))
    return pts, lines


def _try_candidate(pts, lines):
    if len(pts) != 54 or len(lines) != 27:
        return False, 0
    if not _validate(pts, lines):
        return False, 0
    inc = _count_incidents(pts, lines)
    return True, inc


def _best_over_grid(grid, line_specs):
    pts, lines = _build_candidate(grid, line_specs)
    ok, inc = _try_candidate(pts, lines)
    if ok:
        return inc
    return -1


def _optimize():
    best_inc = -1
    best_grid = None
    best_specs = None

    # Grid sizes: (nx, ny) with nx*ny = 54
    # 54 = 2 * 3^3
    # Factors: 1,2,3,6,9,18,27,54
    # (nx, ny) pairs: (1,54),(2,27),(3,18),(6,9),(9,6),(18,3),(27,2),(54,1)
    # We'll try a few of these, especially (6,9) and (9,6) and (3,18) etc.

    # For each grid, we need 27 lines.
    # Available line types and counts:
    # V: nx lines, each with ny points
    # H: ny lines, each with nx points
    # S1: |nx-ny|+1 lines, varying lengths
    # S-1: |nx+ny-1|+1 lines (actually nx+ny-1 diagonals), varying lengths
    # S2, S-2, S1/2, S-1/2: fewer lines, shorter

    # For simplicity and to maximize, we'll use a greedy approach:
    # For a given grid, generate all candidate lines (up to some limit),
    # count their incidences, pick top 27.

    # To keep it fast, we'll limit to a few grid shapes and a limited set of line slopes.

    grid_shapes = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]

    # For each shape, try offset 0..2 for x and y (small range)
    # Actually, offset doesn't matter for incidence count, only relative positions.
    # So we can fix offset to 0.

    for nx, ny in grid_shapes:
        xs = list(range(nx))
        ys = list(range(ny))

        # Generate candidate lines
        candidates = []

        # Vertical lines
        for c in xs:
            candidates.append(('v', c))

        # Horizontal lines
        for c in ys:
            candidates.append(('h', c))

        # Slope 1: x - y + c = 0
        # c = y - x
        min_c1 = min(y - x for x in xs for y in ys)
        max_c1 = max(y - x for x in xs for y in ys)
        for c in range(min_c1, max_c1 + 1):
            candidates.append(('s1', c))

        # Slope -1: x + y - c = 0
        # c = x + y
        min_cn1 = min(x + y for x in xs for y in ys)
        max_cn1 = max(x + y for x in xs for y in ys)
        for c in range(min_cn1, max_cn1 + 1):
            candidates.append(('s-1', c))

        # Slope 2: 2x - y - c = 0
        # c = 2x - y
        min_c2 = min(2 * x - y for x in xs for y in ys)
        max_c2 = max(2 * x - y for x in xs for y in ys)
        for c in range(min_c2, max_c2 + 1):
            candidates.append(('s2', c))

        # Slope -2: 2x + y - c = 0
        # c = 2x + y
        min_cn2 = min(2 * x + y for x in xs for y in ys)
        max_cn2 = max(2 * x + y for x in xs for y in ys)
        for c in range(min_cn2, max_cn2 + 1):
            candidates.append(('s-2', c))

        # Slope 1/2: x - 2y + c = 0
        # c = 2y - x
        min_c12 = min(2 * y - x for x in xs for y in ys)
        max_c12 = max(2 * y - x for x in xs for y in ys)
        for c in range(min_c12, max_c12 + 1):
            candidates.append(('s1/2', c))

        # Slope -1/2: x + 2y - c = 0
        # c = x + 2y
        min_cn12 = min(x + 2 * y for x in xs for y in ys)
        max_cn12 = max(x + 2 * y for x in xs for y in ys)
        for c in range(min_cn12, max_cn12 + 1):
            candidates.append(('s-1/2', c))

        # Now, for each candidate line, compute its incidence count
        pts, _ = _build_candidate((xs, ys), [])
        # Actually, pts is just the grid points
        pt_list = []
        for x in xs:
            for y in ys:
                pt_list.append((x, y))

        line_incidents = []
        seen_norm = set()

        for spec in candidates:
            # Build line
            if spec[0] == 'v':
                c = spec[1]
                A, B, C = 1, 0, -c
            elif spec[0] == 'h':
                c = spec[1]
                A, B, C = 0, 1, -c
            elif spec[0] == 's1':
                c = spec[1]
                A, B, C = 1, -1, c
            elif spec[0] == 's-1':
                c = spec[1]
                A, B, C = 1, 1, -c
            elif spec[0] == 's2':
                c = spec[1]
                A, B, C = 2, -1, -c
            elif spec[0] == 's-2':
                c = spec[1]
                A, B, C = 2, 1, -c
            elif spec[0] == 's1/2':
                c = spec[1]
                A, B, C = 1, -2, c
            elif spec[0] == 's-1/2':
                c = spec[1]
                A, B, C = 1, 2, -c
            else:
                continue

            norm = _normalize_line(A, B, C)
            if norm is None or norm in seen_norm:
                continue
            seen_norm.add(norm)

            # Count incidences
            inc = 0
            for (px, py) in pt_list:
                if A * px + B * py + C == 0:
                    inc += 1
            if inc >= 2:  # Only consider lines with at least 2 points
                line_incidents.append((inc, spec))

        # Sort by incidence count descending
        line_incidents.sort(key=lambda x: -x[0])

        # Take top 27
        if len(line_incidents) >= 27:
            top27 = [spec for _, spec in line_incidents[:27]]
            # Check for duplicates
            norm_set = set()
            valid = True
            for spec in top27:
                if spec[0] == 'v':
                    c = spec[1]
                    A, B, C = 1, 0, -c
                elif spec[0] == 'h':
                    c = spec[1]
                    A, B, C = 0, 1, -c
                elif spec[0] == 's1':
                    c = spec[1]
                    A, B, C = 1, -1, c
                elif spec[0] == 's-1':
                    c = spec[1]
                    A, B, C = 1, 1, -c
                elif spec[0] == 's2':
                    c = spec[1]
                    A, B, C = 2, -1, -c
                elif spec[0] == 's-2':
                    c = spec[1]
                    A, B, C = 2, 1, -c
                elif spec[0] == 's1/2':
                    c = spec[1]
                    A, B, C = 1, -2, c
                elif spec[0] == 's-1/2':
                    c = spec[1]
                    A, B, C = 1, 2, -c
                else:
                    valid = False
                    break
                norm = _normalize_line(A, B, C)
                if norm in norm_set:
                    valid = False
                    break
                norm_set.add(norm)

            if valid:
                total_inc = sum(inc for inc, _ in line_incidents[:27])
                if total_inc > best_inc:
                    best_inc = total_inc
                    best_grid = (xs, ys)
                    best_specs = top27

    if best_grid is None:
        # Fallback to current program's configuration
        return (
            [(x, y) for y in range(1, 10) for x in range(1, 7)],
            [(1, 0, -c) for c in range(1, 7)] +
            [(0, 1, -c) for c in range(1, 10)] +
            [(1, -1, c) for c in range(0, 4)] +
            [(1, 1, -c) for c in range(7, 11)] +
            [(1, -1, c) for c in [-1, 4]] +
            [(1, 1, -c) for c in [6, 11]]
        )

    pts, lines = _build_candidate(best_grid, best_specs)
    return pts, lines


def _cached_result():
    global _RESULT_CACHE
    if not hasattr(_cached_result, '_RESULT_CACHE'):
        _cached_result._RESULT_CACHE = _optimize()
    return _cached_result._RESULT_CACHE


def szemeredi_trotter_incidences():
    pts, lines = _cached_result()
    pts_arr = np.array(pts, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    return pts_arr, lines_arr
# EVOLVE-BLOCK-END
