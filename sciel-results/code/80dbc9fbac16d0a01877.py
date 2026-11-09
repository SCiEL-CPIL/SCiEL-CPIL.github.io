# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from collections import defaultdict


def _count_incidences(points, lines):
    """Count point-line incidences exactly."""
    n, m = len(points), len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def _line_key(A, B, C):
    """Canonical form of a line for deduplication."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _is_valid(points, lines):
    """Check validity constraints."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if not np.all(np.abs(points - np.round(points)) < 1e-6):
        return False
    if not np.all(np.abs(lines - np.round(lines)) < 1e-6):
        return False
    # Check repeated points
    pts = set(tuple(p) for p in points.tolist())
    if len(pts) != 54:
        return False
    # Check lines
    seen_lines = set()
    for row in lines.tolist():
        A, B, C = row
        if A == 0 and B == 0:
            return False
        key = _line_key(A, B, C)
        if key is None:
            return False
        if key in seen_lines:
            return False
        seen_lines.add(key)
    return True


def _initial_parabola_config():
    """
    Initial configuration: points on parabola y = x^2 for x in some range.
    Lines are chords through pairs of these points.
    """
    # Use points on y = x^2, x from -15 to 15 (31 points) plus some extra
    # Actually, let's use a cleaner approach:
    # Place 54 points: use x values that give good chord structure
    
    # Strategy: use points (x, x^2) for x in a set S of size 54
    # Lines through pairs (i, j) in S
    # We want to choose 27 pairs to maximize incidences
    
    # Start with x values from -27 to 26 (54 values)
    x_vals = list(range(-27, 27))
    points = np.array([[x, x * x] for x in x_vals], dtype=np.int64)
    
    # For initial lines, pick 27 random pairs
    random.seed(42)
    pairs = []
    used = set()
    attempts = 0
    while len(pairs) < 27 and attempts < 10000:
        attempts += 1
        i = random.randint(0, 53)
        j = random.randint(0, 53)
        if i != j and (i, j) not in used and (j, i) not in used:
            pairs.append((i, j))
            used.add((i, j))
            used.add((j, i))
    
    # Compute line equations for each pair
    lines = []
    for (i, j) in pairs:
        x1, y1 = x_vals[i], x_vals[i] ** 2
        x2, y2 = x_vals[j], x_vals[j] ** 2
        # Line through (x1, y1) and (x2, y2): 
        # A = y2 - y1, B = x1 - x2, C = x2*y1 - x1*y2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        lines.append([A, B, C])
    
    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _local_search(points, lines, time_limit=580.0, start_time=None):
    """
    Local search: try swapping which pair of points defines each line.
    Also try moving points slightly.
    """
    if start_time is None:
        start_time = time.time()
    
    x_vals = list(range(-27, 27))
    current_inc = _count_incidences(points, lines)
    best_inc = current_inc
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Precompute point coordinates
    pts = points.tolist()
    
    def line_from_pair(i, j):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return [A, B, C]
    
    # Track which pair defines each line
    line_pairs = [(random.randint(0, 53), random.randint(0, 53)) for _ in range(27)]
    # Ensure unique pairs
    seen = set()
    for idx in range(27):
        while True:
            i, j = line_pairs[idx]
            if i != j and (i, j) not in seen and (j, i) not in seen:
                seen.add((i, j))
                seen.add((j, i))
                break
            line_pairs[idx] = (random.randint(0, 53), random.randint(0, 53))
    
    # Rebuild lines from pairs
    for idx in range(27):
        i, j = line_pairs[idx]
        lines[idx] = line_from_pair(i, j)
    
    current_inc = _count_incidences(points, lines)
    best_inc = current_inc
    best_points = points.copy()
    best_lines = lines.copy()
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Try a random move: change one line's pair
        line_idx = random.randint(0, 26)
        old_i, old_j = line_pairs[line_idx]
        
        new_i = random.randint(0, 53)
        new_j = random.randint(0, 53)
        if new_i == new_j:
            continue
        if (new_i, new_j) in seen or (new_j, new_i) in seen:
            # Skip if already used by another line
            # But allow if it's the same line's current pair
            if (new_i, new_j) != (old_i, old_j) and (new_j, new_i) != (old_i, old_j):
                continue
        
        # Remove old pair from seen
        seen.discard((old_i, old_j))
        seen.discard((old_j, old_i))
        # Add new pair
        seen.add((new_i, new_j))
        seen.add((new_j, new_i))
        
        line_pairs[line_idx] = (new_i, new_j)
        lines[line_idx] = line_from_pair(new_i, new_j)
        
        new_inc = _count_incidences(points, lines)
        delta = new_inc - current_inc
        
        if delta > 0 or random.random() < np.exp(delta / max(temp, 0.1)):
            current_inc = new_inc
            if current_inc > best_inc:
                best_inc = current_inc
                best_points = points.copy()
                best_lines = lines.copy()
        else:
            # Revert
            seen.discard((new_i, new_j))
            seen.discard((new_j, new_i))
            seen.add((old_i, old_j))
            seen.add((old_j, old_i))
            line_pairs[line_idx] = (old_i, old_j)
            lines[line_idx] = line_from_pair(old_i, old_j)
        
        temp *= cooling
    
    return best_points, best_lines, best_inc


def _try_concentric_config():
    """
    Alternative: place points in a grid-like pattern and use lines that pass through many.
    Use a (3, 18) grid: 3 columns, 18 rows.
    Lines: vertical lines (3), horizontal lines (18), and some diagonals.
    """
    # Points: x in {0, 1, 2}, y in {0, 1, ..., 17}
    xs = [0, 1, 2]
    ys = list(range(18))
    points_list = [(x, y) for x in xs for y in ys]
    points = np.array(points_list, dtype=np.int64)
    
    lines_list = []
    # 3 vertical lines: x = 0, 1, 2 -> (1, 0, -c)
    for x in xs:
        lines_list.append([1, 0, -x])
    # 18 horizontal lines: y = 0, 1, ..., 17 -> (0, 1, -c)
    for y in ys:
        lines_list.append([0, 1, -y])
    # That's 21 lines. Need 6 more.
    # Add 6 diagonal lines
    # Diagonal: y = x + c, choose c to maximize incidences
    # For x in {0,1,2}, y in {0..17}, y = x + c means y - x = c
    # c ranges from -2 to 17
    # Pick 6 values of c that give good coverage
    for c in [0, 1, 2, 3, 4, 5]:
        lines_list.append([1, -1, -c])  # x - y + c = 0 -> y = x + c
    
    lines = np.array(lines_list, dtype=np.int64)
    return points, lines


def _try_mixed_construction():
    """
    Mixed approach: combine grid structure with strategic lines.
    Use a 6x9 grid (54 points) and choose 27 lines strategically.
    """
    # 6 columns, 9 rows
    xs = list(range(6))
    ys = list(range(9))
    points_list = [(x, y) for x in xs for y in ys]
    points = np.array(points_list, dtype=np.int64)
    
    lines_list = []
    # 6 vertical lines
    for x in xs:
        lines_list.append([1, 0, -x])
    # 9 horizontal lines
    for y in ys:
        lines_list.append([0, 1, -y])
    # That's 15 lines. Need 12 more.
    # Add diagonals with various slopes
    # For slope 1: y = x + c, c in {-5, -4, ..., 8}
    # Each diagonal passes through at most min(6, 9, ...) points
    # For a 6x9 grid, diagonal y = x + c passes through points where y - x = c
    # x in [0,5], y in [0,8], so c in [-5, 8]
    # Number of points on diagonal: depends on c
    # c = 0: (0,0),(1,1),...,(5,5) -> 6 points
    # c = 1: (0,1),...,(5,6) -> 6 points
    # c = -1: (1,0),...,(5,4) -> 5 points
    # etc.
    
    # Pick the best 12 diagonals
    diagonal_candidates = []
    for c in range(-5, 9):
        count = 0
        for x in range(6):
            y = x + c
            if 0 <= y <= 8:
                count += 1
        diagonal_candidates.append((count, c))
    
    diagonal_candidates.sort(reverse=True)
    for count, c in diagonal_candidates[:12]:
        lines_list.append([1, -1, -c])
    
    lines = np.array(lines_list, dtype=np.int64)
    return points, lines


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses multiple construction strategies and local search.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_points = None
    best_lines = None
    best_inc = 0
    
    # Strategy 1: Grid with diagonals
    pts1, lines1 = _try_mixed_construction()
    inc1 = _count_incidences(pts1, lines1)
    if inc1 > best_inc:
        best_inc = inc1
        best_points = pts1.copy()
        best_lines = lines1.copy()
    
    # Strategy 2: Concentric/grid
    pts2, lines2 = _try_concentric_config()
    inc2 = _count_incidences(pts2, lines2)
    if inc2 > best_inc:
        best_inc = inc2
        best_points = pts2.copy()
        best_lines = lines2.copy()
    
    # Strategy 3: Parabola with local search
    if time.time() < deadline - 10:
        pts3, lines3 = _initial_parabola_config()
        inc3 = _count_incidences(pts3, lines3)
        if inc3 > best_inc:
            best_inc = inc3
            best_points = pts3.copy()
            best_lines = lines3.copy()
        
        # Local search on parabola config
        if time.time() < deadline - 5:
            pts3s, lines3s, inc3s = _local_search(pts3, lines3, time_limit=deadline - time.time() - 5)
            if inc3s > best_inc:
                best_inc = inc3s
                best_points = pts3s.copy()
                best_lines = lines3s.copy()
    
    # Strategy 4: Local search on grid config
    if time.time() < deadline - 10:
        # Use grid points, but search over line choices
        # Points fixed as 6x9 grid
        xs = list(range(6))
        ys = list(range(9))
        points_fixed = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
        
        # Random initial lines
        random.seed(123)
        lines_init = []
        for _ in range(27):
            # Random line: pick random A, B, C
            A = random.randint(-5, 5)
            B = random.randint(-5, 5)
            if A == 0 and B == 0:
                A = 1
            C = random.randint(-20, 20)
            lines_init.append([A, B, C])
        lines_init = np.array(lines_init, dtype=np.int64)
        
        # Simple hill climbing on lines
        current_inc = _count_incidences(points_fixed, lines_init)
        best_inc_ls = current_inc
        best_lines_ls = lines_init.copy()
        
        temp = 5.0
        cooling = 0.9995
        
        while time.time() < deadline - 5:
            line_idx = random.randint(0, 26)
            old_line = lines_init[line_idx].copy()
            
            # Perturb the line
            new_A = old_line[0] + random.randint(-3, 3)
            new_B = old_line[1] + random.randint(-3, 3)
            new_C = old_line[2] + random.randint(-5, 5)
            if new_A == 0 and new_B == 0:
                new_A = 1
            
            lines_init[line_idx] = [new_A, new_B, new_C]
            new_inc = _count_incidences(points_fixed, lines_init)
            delta = new_inc - current_inc
            
            if delta > 0 or random.random() < np.exp(delta / max(temp, 0.1)):
                current_inc = new_inc
                if current_inc > best_inc_ls:
                    best_inc_ls = current_inc
                    best_lines_ls = lines_init.copy()
            else:
                lines_init[line_idx] = old_line
            
            temp *= cooling
        
        if best_inc_ls > best_inc:
            best_inc = best_inc_ls
            best_points = points_fixed.copy()
            best_lines = best_lines_ls.copy()
    
    # Validate and return
    if best_points is None or best_lines is None:
        # Fallback to simple grid
        best_points, best_lines = _try_mixed_construction()
    
    # Ensure integer dtype
    best_points = best_points.astype(np.int64)
    best_lines = best_lines.astype(np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
