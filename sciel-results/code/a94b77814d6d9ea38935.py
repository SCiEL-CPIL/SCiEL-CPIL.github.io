import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _count_incidentences(points, lines):
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

def szemeredi_trotter_incidences():
    deadline = time.time() + 600
    safety_margin = 30

    # 6x9 Grid
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    
    lines = []
    seen = set()
    
    def add_line(A, B, C):
        key = _line_key(A, B, C)
        if key not in seen and not (key[0] == 0 and key[1] == 0):
            seen.add(key)
            lines.append(key)

    # Vertical
    for c in xs:
        add_line(1, 0, -c)
    # Horizontal
    for r in ys:
        add_line(0, 1, -r)
    # Diagonals slope 1
    for d in range(-5, 9):
        add_line(1, -1, d)
    # Diagonals slope -1
    for s in range(2, 16):
        add_line(1, 1, -s)
        
    # Add some other slopes to ensure we have enough candidates
    # Slope 2
    for c in range(-10, 10):
        add_line(2, -1, c) # 2x - y + c = 0 -> y = 2x + c
    # Slope 1/2
    for c in range(-10, 10):
        add_line(1, -2, c) # x - 2y + c = 0 -> x = 2y - c

    # Calculate incidences for all candidates
    point_set = set((int(p[0]), int(p[1])) for p in points)
    cand_inc = []
    for (A, B, C) in lines:
        cnt = 0
        for (x, y) in point_set:
            if A * x + B * y + C == 0:
                cnt += 1
        cand_inc.append((cnt, A, B, C))
        
    cand_inc.sort(reverse=True, key=lambda t: t[0])
    top27 = [(A, B, C) for (_, A, B, C) in cand_inc[:27]]
    lines_arr = np.array(top27, dtype=np.int64)
    
    if not _is_valid(points, lines_arr):
        # Fallback
        return points, np.zeros((27,3), dtype=np.int64)

    best_points, best_lines = points, lines_arr
    best_count = _count_incidentences(points, lines_arr)
    
    # Simple local search on lines only (points fixed)
    if time.time() < deadline - safety_margin:
        for _ in range(100):
            if time.time() > deadline - safety_margin:
                break
            # Try replacing a line
            # Generate candidates from pairs of points
            n = len(points)
            best_gain = 0
            best_new_line = None
            best_replace_idx = -1
            
            # Sample candidates
            for _trial in range(20):
                i1 = np.random.randint(n)
                i2 = np.random.randint(n)
                if i1 == i2: continue
                x1, y1 = points[i1]
                x2, y2 = points[i2]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                if A == 0 and B == 0: continue
                key = _line_key(A, B, C)
                if key[0] == 0 and key[1] == 0: continue
                
                # Check if already in lines
                line_keys = [_line_key(int(l[0]), int(l[1]), int(l[2])) for l in best_lines]
                if key in line_keys: continue
                
                # Try replacing each line
                for j in range(27):
                    new_lines = list(best_lines)
                    new_lines[j] = np.array(key, dtype=np.int64)
                    new_lines_arr = np.array(new_lines, dtype=np.int64)
                    if not _is_valid(points, new_lines_arr): continue
                    cnt = _count_incidentences(points, new_lines_arr)
                    gain = cnt - best_count
                    if gain > best_gain:
                        best_gain = gain
                        best_new_line = key
                        best_replace_idx = j
                        
            if best_gain > 0:
                best_lines[best_replace_idx] = np.array(best_new_line, dtype=np.int64)
                best_count += best_gain

    return best_points, best_lines
