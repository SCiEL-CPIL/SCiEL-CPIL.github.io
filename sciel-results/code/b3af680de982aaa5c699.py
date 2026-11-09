# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd


def _count_incidences(points, lines):
    """Count point-line incidences exactly using int64 arithmetic."""
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


def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|) = 1 and first nonzero is positive."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)


def _is_valid(points, lines):
    """Check if the configuration is valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check points are integers
    if not np.all(np.isclose(points, np.round(points), atol=1e-6)):
        return False
    points_int = np.round(points).astype(np.int64)
    
    # Check no repeated points
    if len(set(map(tuple, points_int.tolist()))) != 54:
        return False
    
    # Check lines
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not all(np.isclose(v, np.round(v), atol=1e-6) for v in [A, B, C]):
            return False
        A, B, C = int(np.round(A)), int(np.round(B)), int(np.round(C))
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in line_set:
            return False
        line_set.add(norm)
    
    return True


def _baseline():
    """Original baseline construction."""
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
    return points, lines


def _grid_construction(cols, rows_count):
    """Grid construction with cols x rows_count points."""
    xs = np.arange(1, cols + 1)
    ys = np.arange(1, rows_count + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    # Horizontal lines
    for y in range(1, rows_count + 1):
        lines.append((0, 1, -y))
    # Vertical lines
    for x in range(1, cols + 1):
        lines.append((1, 0, -x))
    # Diagonal lines with slope 1
    for d in range(1 - cols + 1, rows_count):
        # y = x + d -> x - y + d = 0
        # Check if any points lie on this line
        has_points = False
        for x in range(1, cols + 1):
            y = x + d
            if 1 <= y <= rows_count:
                has_points = True
                break
        if has_points:
            lines.append((1, -1, d))
    
    # Take first 27 lines
    lines = lines[:27]
    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _local_search(points, lines, max_iters=10000, deadline=None):
    """Local search to improve incidence count."""
    points = points.copy()
    lines = lines.copy()
    best_count = _count_incidences(points, lines)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try moving a random point
        i = random.randint(0, len(points) - 1)
        orig_x, orig_y = points[i]
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
        
        new_x, new_y = orig_x + dx, orig_y + dy
        
        # Check if this creates a duplicate point
        points[i] = [new_x, new_y]
        if len(set(map(tuple, points.tolist()))) != len(points):
            points[i] = [orig_x, orig_y]
            continue
        
        new_count = _count_incidences(points, lines)
        if new_count > best_count:
            best_count = new_count
        else:
            points[i] = [orig_x, orig_y]
        
        # Try changing a random line
        j = random.randint(0, len(lines) - 1)
        orig_A, orig_B, orig_C = lines[j]
        
        # Generate a new random line through 2 random points
        if len(points) >= 2:
            p1_idx = random.randint(0, len(points) - 1)
            p2_idx = random.randint(0, len(points) - 1)
            if p1_idx != p2_idx:
                x1, y1 = points[p1_idx]
                x2, y2 = points[p2_idx]
                A = y2 - y1
                B = x1 - x2
                C = -(A * x1 + B * y1)
                if A != 0 or B != 0:
                    # Normalize
                    g = gcd(abs(A), gcd(abs(B), abs(C)))
                    if g > 0:
                        A //= g
                        B //= g
                        C //= g
                    lines[j] = [A, B, C]
                    
                    # Check for duplicate lines
                    line_set = set()
                    valid = True
                    for l in lines:
                        Al, Bl, Cl = int(l[0]), int(l[1]), int(l[2])
                        if Al == 0 and Bl == 0:
                            valid = False
                            break
                        norm = _normalize_line(Al, Bl, Cl)
                        if norm in line_set:
                            valid = False
                            break
                        line_set.add(norm)
                    
                    if valid:
                        new_count = _count_incidences(points, lines)
                        if new_count > best_count:
                            best_count = new_count
                        else:
                            lines[j] = [orig_A, orig_B, orig_C]
                    else:
                        lines[j] = [orig_A, orig_B, orig_C]
    
    return points, lines, best_count


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    """
    deadline = time.time() + 580  # Leave 20s safety margin
    
    # Start with baseline
    points, lines = _baseline()
    best_points, best_lines = points.copy(), lines.copy()
    best_count = _count_incidences(points, lines)
    
    # Try different grid constructions
    for cols in range(3, 10):
        if time.time() > deadline:
            break
        rows_count = 54 // cols
        if cols * rows_count != 54:
            continue
        pts, lns = _grid_construction(cols, rows_count)
        if _is_valid(pts, lns):
            cnt = _count_incidences(pts, lns)
            if cnt > best_count:
                best_count = cnt
                best_points, best_lines = pts.copy(), lns.copy()
    
    # Local search from best configuration
    if time.time() < deadline - 60:
        pts, lns, cnt = _local_search(best_points, best_lines, max_iters=50000, deadline=deadline - 30)
        if cnt > best_count:
            best_count = cnt
            best_points, best_lines = pts.copy(), lns.copy()
    
    # Ensure valid output
    best_points = np.round(best_points).astype(np.int64)
    best_lines = np.round(best_lines).astype(np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
