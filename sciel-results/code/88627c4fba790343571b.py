import numpy as np
import time
import math
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_two_points(p1, p2):
    """Return normalized line through two points, or None if same point."""
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

def _validate(points, lines):
    """Check validity of the configuration."""
    if len(points) != 54 or len(lines) != 27:
        return False
    point_set = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))):
            return False
        if x != int(x) or y != int(y):
            return False
        if (x, y) in point_set:
            return False
        point_set.add((x, y))
    line_set = set()
    for l in lines:
        if len(l) != 3:
            return False
        A, B, C = l
        if not (isinstance(A, (int, np.integer)) and isinstance(B, (int, np.integer)) and isinstance(C, (int, np.integer))):
            return False
        if A != int(A) or B != int(B) or C != int(C):
            return False
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(int(A), int(B), int(C))
        if norm in line_set:
            return False
        line_set.add(norm)
    return True

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 620  # safety margin
    
    # Initial configuration: 6x9 grid
    # Points: {0,...,5} x {0,...,8}
    points = []
    for x in range(6):
        for y in range(9):
            points.append([x, y])
    points = np.array(points, dtype=np.int64)
    
    # Initial lines: all vertical and horizontal lines of the grid
    lines = []
    for x in range(6):
        lines.append([1, 0, -x])  # x = const
    for y in range(9):
        lines.append([0, 1, -y])  # y = const
    # That's 15 lines, need 27. Add some diagonal lines.
    # For now, just use these 15 and pad with more diagonals later.
    
    # Let me restart with a better initial config
    # Use the Szemeredi-Trotter grid: k=3, points {1..3}x{1..18}, lines y=ax+b for a in {1,2,3}, b in {1..9}
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append([a, -1, b])
    lines = np.array(rows, dtype=np.int64)
    
    best_score = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Local search
    # For each point, try moving it to a nearby position
    # For each line, try replacing it with a line through two other points
    
    # Precompute incidence matrix
    def compute_incidence_matrix(points, lines):
        n = len(points)
        m = len(lines)
        inc = np.zeros((n, m), dtype=np.int64)
        for i in range(n):
            x, y = points[i]
            for j in range(m):
                A, B, C = lines[j]
                if A * x + B * y + C == 0:
                    inc[i, j] = 1
        return inc
    
    # Use simulated annealing style local search
    current_score = best_score
    current_points = best_points.copy()
    current_lines = best_lines.copy()
    
    # Helper: try moving a point
    def try_move_point(points, lines, pi, new_pos):
        """Return new score if we move point pi to new_pos, or -1 if invalid."""
        x, y = new_pos
        if x == points[pi][0] and y == points[pi][1]:
            return current_score  # no change
        # Check if new position is already taken
        for i in range(len(points)):
            if i != pi and points[i][0] == x and points[i][1] == y:
                return -1
        # Compute change in incidences
        old_x, old_y = points[pi]
        delta = 0
        for j in range(len(lines)):
            A, B, C = lines[j]
            old_inc = 1 if (A * old_x + B * old_y + C == 0) else 0
            new_inc = 1 if (A * x + B * y + C == 0) else 0
            delta += new_inc - old_inc
        return current_score + delta
    
    # Helper: try replacing a line
    def try_replace_line(points, lines, lj, new_line):
        """Return new score if we replace line lj with new_line, or -1 if invalid."""
        A, B, C = new_line
        if A == 0 and B == 0:
            return -1
        # Check if new line is same as any existing line
        norm_new = _normalize_line(int(A), int(B), int(C))
        for j in range(len(lines)):
            if j == lj:
                continue
            A2, B2, C2 = lines[j]
            norm_old = _normalize_line(int(A2), int(B2), int(C2))
            if norm_new == norm_old:
                return -1
        # Compute change in incidences
        old_A, old_B, old_C = lines[lj]
        delta = 0
        for i in range(len(points)):
            x, y = points[i]
            old_inc = 1 if (old_A * x + old_B * y + old_C == 0) else 0
            new_inc = 1 if (A * x + B * y + C == 0) else 0
            delta += new_inc - old_inc
        return current_score + delta
    
    # Local search iterations
    max_iter = 500000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Randomly choose: move a point or replace a line
        if np.random.random() < 0.5:
            # Move a point
            pi = np.random.randint(len(current_points))
            old_x, old_y = current_points[pi]
            # Try nearby positions
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            new_pos = (old_x + dx, old_y + dy)
            new_score = try_move_point(current_points, current_lines, pi, new_pos)
            if new_score > 0 and new_score > current_score:
                current_points[pi] = new_pos
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
        else:
            # Replace a line with one through two random other points
            lj = np.random.randint(len(current_lines))
            # Pick two random points
            i1 = np.random.randint(len(current_points))
            i2 = np.random.randint(len(current_points))
            if i1 == i2:
                continue
            p1 = tuple(current_points[i1])
            p2 = tuple(current_points[i2])
            new_line = _line_from_two_points(p1, p2)
            if new_line is None:
                continue
            new_score = try_replace_line(current_points, current_lines, lj, new_line)
            if new_score > 0 and new_score > current_score:
                current_lines[lj] = new_line
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
    
    # Final validation
    if not _validate(best_points, best_lines):
        # Fallback to initial config
        return points, lines
    
    return best_points, best_lines
