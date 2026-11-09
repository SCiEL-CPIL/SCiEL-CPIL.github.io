# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
import random

def _count_incidentences(points, lines):
    """Count (point, line) pairs where A*x + B*y + C == 0 exactly using vectorization."""
    # points: (54, 2), lines: (27, 3)
    # Expand lines to (27, 2) for A, B and (27,) for C
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    # x: (54, 1), y: (54, 1)
    x = points[:, 0:1]
    y = points[:, 1:2]
    
    # Compute A*x + B*y + C for all pairs: (54, 27)
    val = A[None, :] * x + B[None, :] * y + C[None, :]
    
    # Count zeros
    return np.sum(val == 0)

def _normalize_line(A, B, C):
    """Normalize (A, B, C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
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
    """Check all validity constraints."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check points: no NaN/inf, integer-valued, no repeats
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
        
    # Check lines: no A=B=0, no repeats after normalization
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

def _repair(points, lines):
    """
    Repair points and lines to satisfy validity constraints.
    Returns repaired (points, lines) arrays.
    """
    points = points.copy()
    lines = lines.copy()
    
    # 1. Round to integers
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    # 2. Fix duplicate points
    unique_pts = []
    seen = set()
    for i in range(54):
        p = tuple(points[i])
        if p in seen:
            # Move to a random nearby location
            while True:
                nx = points[i][0] + random.randint(-5, 5)
                ny = points[i][1] + random.randint(-5, 5)
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    points[i] = [nx, ny]
                    break
        else:
            seen.add(p)
            
    # 3. Fix lines: A=B=0 or duplicates
    unique_lines = []
    seen_lines = set()
    for i in range(27):
        A, B, C = lines[i]
        if A == 0 and B == 0:
            # Assign a random line through two distinct points
            while True:
                i1 = random.randint(0, 53)
                i2 = random.randint(0, 53)
                if i1 != i2:
                    x1, y1 = points[i1]
                    x2, y2 = points[i2]
                    A_new = y2 - y1
                    B_new = x1 - x2
                    C_new = x2 * y1 - x1 * y2
                    if A_new != 0 or B_new != 0:
                        key = _line_key(A_new, B_new, C_new)
                        if key not in seen_lines:
                            seen_lines.add(key)
                            lines[i] = [key[0], key[1], key[2]]
                            break
        else:
            key = _line_key(int(A), int(B), int(C))
            if key in seen_lines:
                # Find a new unique line
                replaced = False
                for _ in range(100):
                    i1 = random.randint(0, 53)
                    i2 = random.randint(0, 53)
                    if i1 != i2:
                        x1, y1 = points[i1]
                        x2, y2 = points[i2]
                        A_new = y2 - y1
                        B_new = x1 - x2
                        C_new = x2 * y1 - x1 * y2
                        if A_new != 0 or B_new != 0:
                            new_key = _line_key(A_new, B_new, C_new)
                            if new_key not in seen_lines:
                                seen_lines.discard(key)
                                seen_lines.add(new_key)
                                lines[i] = [new_key[0], new_key[1], new_key[2]]
                                replaced = True
                                break
                if not replaced:
                    # Fallback: force a unique line
                    seen_lines.discard(key)
                    # Use a line far away
                    A_new = 1
                    B_new = 0
                    C_new = random.randint(-100, 100)
                    new_key = _line_key(A_new, B_new, C_new)
                    while new_key in seen_lines:
                        C_new += 1
                        new_key = _line_key(A_new, B_new, C_new)
                    seen_lines.add(new_key)
                    lines[i] = [new_key[0], new_key[1], new_key[2]]
            else:
                seen_lines.add(key)
                
    return points, lines

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences using Simulated Annealing on integer lattice.
    """
    deadline = time.time() + 600
    safety_margin = 10
    
    # Initial configuration: Grid 6x9
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    
    # Initial lines: Top 27 from grid candidates
    lines_list = []
    seen = set()
    def add_line(A, B, C):
        key = _line_key(A, B, C)
        if key not in seen and not (key[0] == 0 and key[1] == 0):
            seen.add(key)
            lines_list.append(key)
            
    for c in xs: add_line(1, 0, -c)
    for r in ys: add_line(0, 1, -r)
    for d in range(-5, 9): add_line(1, -1, d)
    for s in range(2, 16): add_line(1, 1, -s)
    
    # Sort by incidence
    point_set = set((int(p[0]), int(p[1])) for p in points)
    cand_inc = []
    for (A, B, C) in lines_list:
        cnt = sum(1 for (x, y) in point_set if A * x + B * y + C == 0)
        cand_inc.append((cnt, A, B, C))
    cand_inc.sort(reverse=True, key=lambda t: t[0])
    lines = np.array(cand_inc[:27], dtype=np.int64)
    
    best_points, best_lines = points.copy(), lines.copy()
    best_count = _count_incidentences(best_points, best_lines)
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline - safety_margin and T > T_min:
        # Generate a move
        move_type = random.random()
        
        # Copy current state
        cur_points = best_points.copy()
        cur_lines = best_lines.copy()
        
        if move_type < 0.6:
            # Move a point
            idx = random.randint(0, 53)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            cur_points[idx][0] += dx
            cur_points[idx][1] += dy
        else:
            # Move a line (replace with line through two random points)
            idx = random.randint(0, 26)
            i1 = random.randint(0, 53)
            i2 = random.randint(0, 53)
            if i1 != i2:
                x1, y1 = best_points[i1]
                x2, y2 = best_points[i2]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                if A != 0 or B != 0:
                    key = _line_key(A, B, C)
                    cur_lines[idx] = key
        
        # Calculate new score
        new_count = _count_incidentences(cur_points, cur_lines)
        
        # Acceptance criterion
        delta = new_count - best_count
        if delta > 0 or random.random() < np.exp(delta / T):
            best_points = cur_points
            best_lines = cur_lines
            best_count = new_count
            
        T *= cooling
        
        # Periodic repair to maintain validity (though SA keeps integers, duplicates might occur)
        if random.random() < 0.05:
            best_points, best_lines = _repair(best_points, best_lines)
            best_count = _count_incidentences(best_points, best_lines)
            
    # Final repair to ensure validity
    best_points, best_lines = _repair(best_points, best_lines)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
